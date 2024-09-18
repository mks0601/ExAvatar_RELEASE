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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import corresponding_points_alignment
import json
import math

class Custom(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.root_path = osp.join('..', 'data', 'Custom', 'data', cfg.subject_id)
        self.transform = transform
        self.kpt = {
                    'num': 133,
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }
        self.cam_params, self.img_paths, self.kpts, self.smplx_params, self.flame_params, self.flame_shape_param, self.frame_idx_list = self.load_data()
        self.get_smplx_trans_init() # get initial smplx translation 
        self.get_flame_root_init() # get initial flame root pose and translation

    def load_data(self):
        cam_params, img_paths, kpts, smplx_params, flame_params = {}, {}, {}, {}, {}
        frame_idx_list = []

        # read frame idxs
        with open(osp.join(self.root_path, 'frame_list_all.txt')) as f:
            lines = f.readlines()
        for frame_idx in lines:
            frame_idx_list.append(int(frame_idx))
        
        # check if camera parameters from COLMAP are available
        if osp.isfile(osp.join(self.root_path, 'sprase', 'cameras.txt')) and osp.isfile(osp.join(self.root_path, 'sprase', 'images.txt')):
            cam_params_from_colmap = True
            with open(osp.join(self.root_path, 'sparse', 'cameras.txt')) as f:
                lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                splitted = line.split()
                _, _, width, height, focal_x, focal_y, princpt_x, princpt_y = splitted
            focal = np.array((float(focal_x), float(focal_y)), dtype=np.float32) # shared across all frames
            princpt = np.array((float(princpt_x), float(princpt_y)), dtype=np.float32) # shared across all frames
            with open(osp.join(self.root_path, 'sparse', 'images.txt')) as f:
                lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                if 'png' not in line:
                    continue
                splitted = line.split()
                frame_idx = int(splitted[-1][:-4])
                qw, qx, qy, qz = float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])
                tx, ty, tz = float(splitted[5]), float(splitted[6]), float(splitted[7])
                R = quaternion_to_matrix(torch.FloatTensor([qw, qx, qy, qz])).numpy()
                t = np.array([tx, ty, tz], dtype=np.float32)
                cam_params[frame_idx] = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}
        else:
            cam_params_from_colmap = False
       
        for i in range(len(frame_idx_list)):
            frame_idx = frame_idx_list[i]

            # load camera parameters
            if not cam_params_from_colmap:
                cam_param_path = osp.join(self.root_path, 'cam_params', str(frame_idx) + '.json')
                with open(cam_param_path) as f:
                    cam_params[frame_idx] = json.load(f)

            # load image paths
            img_path = osp.join(self.root_path, 'frames', str(frame_idx) + '.png')
            img_paths[frame_idx] = img_path

            # load keypoints
            kpt_path = osp.join(self.root_path, 'keypoints_whole_body', str(frame_idx) + '.json')
            with open(kpt_path) as f:
                kpt = np.array(json.load(f), dtype=np.float32)
            kpt = change_kpt_name(kpt, self.kpt['name'], smpl_x.kpt['name'])
            kpts[frame_idx] = kpt
            
            # load initial smplx parameters
            smplx_param_path = osp.join(self.root_path, 'smplx_init', str(frame_idx) + '.json')
            with open(smplx_param_path) as f:
                smplx_params[frame_idx] = {k: torch.FloatTensor(v) for k,v in json.load(f).items()}

            # load initial flame parameters
            flame_param_path = osp.join(self.root_path, 'flame_init', 'flame_params', str(frame_idx) + '.json')
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

        with open(osp.join(self.root_path, 'flame_init', 'shape_param.json')) as f:
            flame_shape_param = torch.FloatTensor(json.load(f))
        return cam_params, img_paths, kpts, smplx_params, flame_params, flame_shape_param, frame_idx_list
    
    def get_smplx_trans_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            cam_param = self.cam_params[frame_idx]
            focal, princpt = cam_param['focal'], cam_param['princpt']

            kpt = self.kpts[frame_idx]
            kpt_img = kpt[:,:2]
            kpt_valid = (kpt[:,2:] > 0.2).astype(np.float32)
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
            cam_param = self.cam_params[frame_idx]
            focal, princpt = cam_param['focal'], cam_param['princpt']
            
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
        kpt = self.kpts[frame_idx]
        kpt_img = kpt[:,:2]
        kpt_valid = (kpt[:,2:] > 0.2).astype(np.float32)

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
        cam_param = self.cam_params[frame_idx]
        focal = np.array(cam_param['focal'], dtype=np.float32)
        princpt = np.array(cam_param['princpt'], dtype=np.float32)
        focal_proj = np.array([focal[0] / bbox[2] * cfg.proj_shape[1], focal[1] / bbox[3] * cfg.proj_shape[0]], dtype=np.float32)
        focal_face = np.array([focal[0] / bbox_face[2] * cfg.face_img_shape[1], focal[1] / bbox_face[3] * cfg.face_img_shape[0]], dtype=np.float32)
        princpt_proj = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.proj_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.proj_shape[1]], dtype=np.float32)
        princpt_face = np.array([(princpt[0] - bbox_face[0]) / bbox_face[2] * cfg.face_img_shape[1], (princpt[1] - bbox_face[1]) / bbox_face[3] * cfg.face_img_shape[1]], dtype=np.float32)

        data = {'img_face': img_face, 'kpt_img': kpt_img, 'kpt_valid': kpt_valid, 'smplx_param': smplx_param, 'flame_param': flame_param, 'flame_valid': flame_valid, 'cam_param': {'focal': focal, 'princpt': princpt}, 'cam_param_proj': {'focal': focal_proj, 'princpt': princpt_proj}, 'cam_param_face': {'focal': focal_face, 'princpt': princpt_face}, 'frame_idx': frame_idx, 'img_orig': img_orig}
        return data


