import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox, set_aspect_ratio, get_patch_img
from utils.transforms import change_kpt_name
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from pytorch3d.ops import corresponding_points_alignment
import json
import math
import pickle

class XHumans(torch.utils.data.Dataset):
    def __init__(self, transform):
        if cfg.subject_id == '00028':
            self.capture_id = 'Take15'
        elif cfg.subject_id == '00034':
            self.capture_id = 'Take14'
        elif cfg.subject_id == '00087':
            self.capture_id = 'Take8'
        self.root_path = osp.join('..', 'data', 'XHumans', 'data', cfg.subject_id, 'train', self.capture_id)
        self.transform = transform
        self.kpt = {
                    'num': 133,
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }
        self.img_paths, self.kpts, self.smplx_params, self.flame_params, self.flame_shape_param, self.cam_params, self.frame_idx_list = self.load_data()
        self.get_smplx_trans_init() # get initial smplx translation 
        self.get_flame_root_init() # get initial flame root pose and translation

    def load_data(self):
        
        # load image paths
        img_paths = {}
        img_path_list = glob(osp.join(self.root_path, 'render', 'image', '*.png'))
        for img_path in img_path_list:
            frame_idx = int(img_path.split('/')[-1].split('_')[1][:-4])
            img_paths[frame_idx] = img_path

        # load keypoints
        kpts = {}
        kpt_path_list = glob(osp.join(self.root_path, 'render', 'keypoints_whole_body', '*.json'))
        for kpt_path in kpt_path_list:
            frame_idx = int(kpt_path.split('/')[-1][:-5])
            with open(kpt_path) as f:
                kpt = np.array(json.load(f), dtype=np.float32)
            kpt = change_kpt_name(kpt, self.kpt['name'], smpl_x.kpt['name'])
            kpts[frame_idx] = kpt

        # load initial flame parameters
        flame_params = {}
        flame_param_path_list = glob(osp.join(self.root_path, 'render', 'flame_init', 'flame_params', '*.json'))
        for flame_param_path in flame_param_path_list:
            frame_idx = int(flame_param_path.split('/')[-1][:-5])
            with open(flame_param_path) as f:
                flame_param = json.load(f)
            if not flame_param['is_valid']:
                for k in flame_param.keys():
                    if 'pose' in k:
                        flame_param[k] = torch.zeros((3)).float() # dummy
                flame_param['expr'] = torch.zeros((flame.expr_param_dim)).float() # dummy
            for k,v in flame_param.items():
                if k == 'is_valid':
                    continue
                else:
                    flame_param[k] = torch.FloatTensor(v)
            flame_params[frame_idx] = flame_param
        with open(osp.join(self.root_path, 'render', 'flame_init', 'shape_param.json')) as f:
            flame_shape_param = torch.FloatTensor(json.load(f))

        # load smplx parameters
        smplx_params = {}
        smplx_param_path_list = glob(osp.join(self.root_path, 'SMPLX', '*.pkl'))
        for smplx_param_path in smplx_param_path_list:
            frame_idx = int(smplx_param_path.split('/')[-1].split('-')[1].split('_')[0][1:])
            with open(smplx_param_path, 'rb') as f:
                smplx_param = pickle.load(f, encoding='latin1')
            smplx_params[frame_idx] = {'root_pose': smplx_param['global_orient'], \
                                                    'body_pose': smplx_param['body_pose'].reshape(-1,3), \
                                                    'jaw_pose': smplx_param['jaw_pose'], \
                                                    'leye_pose': smplx_param['leye_pose'], \
                                                    'reye_pose': smplx_param['reye_pose'], \
                                                    'lhand_pose': smplx_param['left_hand_pose'].reshape(-1,3), \
                                                    'rhand_pose': smplx_param['right_hand_pose'].reshape(-1,3), \
                                                    'expr': flame_params[frame_idx]['expr'].numpy()} # use flame's one
            smplx_params[frame_idx] = {k: torch.FloatTensor(v) for k,v in smplx_params[frame_idx].items()}

        # load cameras
        cam_params = {}
        cam_param = dict(np.load(osp.join(self.root_path, 'render', 'cameras.npz'), allow_pickle=True))
        focal = np.array([cam_param['intrinsic'][0][0], cam_param['intrinsic'][1][1]], dtype=np.float32)
        princpt = np.array([cam_param['intrinsic'][0][2], cam_param['intrinsic'][1][2]], dtype=np.float32)
        R, t = cam_param['extrinsic'][:,:3,:3].astype(np.float32), cam_param['extrinsic'][:,:3,3].astype(np.float32)
        assert len(R) == len(t)
        for i, frame_idx in enumerate(sorted(list(img_paths.keys()))):
            cam_params[frame_idx] = {'focal': focal, 'princpt': princpt, 'R': R[i], 't': t[i]}

            # world coordinate -> camera coordinate
            root_pose = axis_angle_to_matrix(torch.FloatTensor(smplx_params[frame_idx]['root_pose']))
            cam_R = torch.FloatTensor(R[i])
            root_pose = matrix_to_axis_angle(torch.matmul(cam_R, root_pose))
            smplx_params[frame_idx]['root_pose'] = root_pose

        frame_idx_list = []
        for frame_idx in img_paths.keys():
            frame_idx_list.append(frame_idx)
	
        return img_paths, kpts, smplx_params, flame_params, flame_shape_param, cam_params, frame_idx_list

    def get_smplx_trans_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            focal, princpt = self.cam_params[frame_idx]['focal'], self.cam_params[frame_idx]['princpt']

            kpt = self.kpts[frame_idx]
            kpt_img = kpt[:,:2]
            kpt_valid = (kpt[:,2:] > 0.5).astype(np.float32)
            bbox = get_bbox(kpt_img, kpt_valid[:,0])
            bbox = set_aspect_ratio(bbox)

            t_z = math.sqrt(focal[0]*focal[1]*cfg.body_3d_size*cfg.body_3d_size/(bbox[2]*bbox[3])) # meter
            t_x = bbox[0] + bbox[2]/2 # pixel
            t_y = bbox[1] + bbox[3]/2 # pixel
            t_x = (t_x - princpt[0]) / focal[0] * t_z # meter
            t_y = (t_y - princpt[1]) / focal[1] * t_z # meter
            t_xyz = torch.FloatTensor([t_x, t_y, t_z]) 
            self.smplx_params[frame_idx]['trans'] = t_xyz

    def get_flame_root_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            focal, princpt = self.cam_params[frame_idx]['focal'], self.cam_params[frame_idx]['princpt']
            
            smplx_root_pose = axis_angle_to_matrix(self.smplx_params[frame_idx]['root_pose'])
            smplx_root_trans = self.smplx_params[frame_idx]['trans']
            smplx_init = torch.matmul(smplx_root_pose, smpl_x.layer.v_template.permute(1,0)).permute(1,0)
            smplx_init = (smplx_init - smplx_init.mean(0)[None,:] + smplx_root_trans[None,:])[smpl_x.face_vertex_idx,:]
            
            # get initial root pose and translation with the rigid alignment
            flame_init = flame.layer.v_template
            RTs = corresponding_points_alignment(flame_init[None], smplx_init[None])
            R = RTs.R.permute(0,2,1)[0]
            flame_init = torch.matmul(R, flame_init.permute(1,0)).permute(1,0)
            self.flame_params[frame_idx]['root_pose'] = matrix_to_axis_angle(R)
            self.flame_params[frame_idx]['trans'] = -flame_init.mean(0) + smplx_init.mean(0)

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_idx_list[idx]

        # 2D keypoint
        kpt_img = self.kpts[frame_idx][:,:2]
        kpt_valid = (self.kpts[frame_idx][:,2:] > 0.5).astype(np.float32)

        # load image
        img_orig = load_img(self.img_paths[frame_idx])
        img_height, img_width = img_orig.shape[0], img_orig.shape[1]
        bbox = get_bbox(kpt_img, kpt_valid[:,0])
        bbox = set_aspect_ratio(bbox)
        if np.sum(kpt_valid[smpl_x.kpt['part_idx']['face'],0]) == 0:
            self.flame_params[frame_idx]['is_valid'] = False
            bbox_face = np.array([0,0,1,1], dtype=np.float32)
        else:
            bbox_face = get_bbox(kpt_img[smpl_x.kpt['part_idx']['face'],:], kpt_valid[smpl_x.kpt['part_idx']['face'],0])
        bbox_face = set_aspect_ratio(bbox_face)
        img_face, _, _ = get_patch_img(img_orig, bbox_face, cfg.face_img_shape)
        img_face = self.transform(img_face.astype(np.float32))/255.

        # keypoint affine transformation
        _, img2bb_trans, bb2img_trans = get_patch_img(img_orig, bbox, cfg.proj_shape)
        kpt_img_xy1 = np.concatenate((kpt_img, np.ones_like(kpt_img[:,:1])),1)
        kpt_img = np.dot(img2bb_trans, kpt_img_xy1.transpose(1,0)).transpose(1,0)
        
        # smplx parameter
        smplx_param = self.smplx_params[frame_idx]

        # flame parameter
        flame_param = self.flame_params[frame_idx]
        flame_param['shape'] = self.flame_shape_param
        flame_valid = flame_param['is_valid']

        # modify intrincis to 1) directly project 3D coordinates to cfg.proj_shape space and 2) directly unwrap cfg.face_img_shape face images to UV space
        focal = self.cam_params[frame_idx]['focal']
        princpt = self.cam_params[frame_idx]['princpt']
        focal_proj = np.array([focal[0] / bbox[2] * cfg.proj_shape[1], focal[1] / bbox[3] * cfg.proj_shape[0]], dtype=np.float32)
        focal_face = np.array([focal[0] / bbox_face[2] * cfg.face_img_shape[1], focal[1] / bbox_face[3] * cfg.face_img_shape[0]], dtype=np.float32)
        princpt_proj = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.proj_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.proj_shape[1]], dtype=np.float32)
        princpt_face = np.array([(princpt[0] - bbox_face[0]) / bbox_face[2] * cfg.face_img_shape[1], (princpt[1] - bbox_face[1]) / bbox_face[3] * cfg.face_img_shape[1]], dtype=np.float32)

        data = {'img_face': img_face, 'kpt_img': kpt_img, 'kpt_valid': kpt_valid, 'smplx_param': smplx_param, 'flame_param': flame_param, 'flame_valid': flame_valid, 'cam_param': {'focal': focal, 'princpt': princpt}, 'cam_param_proj': {'focal': focal_proj, 'princpt': princpt_proj}, 'cam_param_face': {'focal': focal_face, 'princpt': princpt_face}, 'frame_idx': frame_idx, 'img_orig': img_orig}
        return data



