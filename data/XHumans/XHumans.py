import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox
from utils.transforms import transform_joint_to_other_db
import pickle
import json

class XHumans(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.data_split = data_split
        self.root_path = osp.join('..', 'data', 'XHumans', 'data', cfg.subject_id)
        self.transform = transform
        self.img_paths, self.depthmap_paths, self.smplx_params, self.cam_params, self.frame_idx_list = self.load_data()
        self.load_id_info()

    def load_data(self):
        img_paths, depthmap_paths, smplx_params, cam_params, frame_idx_list = {}, {}, {}, {}, []
        
        if cfg.fit_pose_to_test:
            data_split = 'test'
        else:
            data_split = self.data_split

        capture_path_list = glob(osp.join(self.root_path, data_split, '*'))
        for capture_path in capture_path_list:
            capture_id = capture_path.split('/')[-1]
            
            # load image paths
            img_paths[capture_id] = {}
            img_path_list = glob(osp.join(capture_path, 'render', 'image', '*.png'))
            for img_path in img_path_list:
                frame_idx = int(img_path.split('/')[-1].split('_')[1][:-4])
                img_paths[capture_id][frame_idx] = img_path

            # load depthmap paths
            depthmap_paths[capture_id] = {}
            depthmap_path_list = glob(osp.join(capture_path, 'render', 'depth', '*.tiff'))
            for depthmap_path in depthmap_path_list:
                frame_idx = int(depthmap_path.split('/')[-1].split('_')[1][:-5])
                depthmap_paths[capture_id][frame_idx] = depthmap_path

            # load smplx parameters
            smplx_params[capture_id] = {}
            smplx_param_path_list = glob(osp.join(capture_path, 'SMPLX', '*.pkl'))
            for smplx_param_path in smplx_param_path_list:
                frame_idx = int(smplx_param_path.split('/')[-1].split('-')[1].split('_')[0][1:])
                with open(smplx_param_path, 'rb') as f:
                    smplx_param = pickle.load(f, encoding='latin1')
                with open(osp.join(capture_path, 'render', 'flame_init', 'flame_params', '%06d.json' % frame_idx)) as f:
                    flame_param = json.load(f)
                if flame_param['is_valid']:
                    expr = np.array(flame_param['expr'], dtype=np.float32)
                else:
                    expr = np.zeros((flame.expr_param_dim), dtype=np.float32) # dummy
                smplx_params[capture_id][frame_idx] = {'root_pose': smplx_param['global_orient'], \
                                                        'body_pose': smplx_param['body_pose'].reshape(-1,3), \
                                                        'jaw_pose': smplx_param['jaw_pose'], \
                                                        'leye_pose': smplx_param['leye_pose'], \
                                                        'reye_pose': smplx_param['reye_pose'], \
                                                        'lhand_pose': smplx_param['left_hand_pose'].reshape(-1,3), \
                                                        'rhand_pose': smplx_param['right_hand_pose'].reshape(-1,3), \
                                                        'expr': expr, # use flame's one
                                                        'trans': smplx_param['transl']}
                smplx_params[capture_id][frame_idx] = {k: torch.FloatTensor(v) for k,v in smplx_params[capture_id][frame_idx].items()}

            # load cameras
            cam_params[capture_id] = {}
            cam_param = dict(np.load(osp.join(capture_path, 'render', 'cameras.npz'), allow_pickle=True))
            focal = np.array([cam_param['intrinsic'][0][0], cam_param['intrinsic'][1][1]], dtype=np.float32)
            princpt = np.array([cam_param['intrinsic'][0][2], cam_param['intrinsic'][1][2]], dtype=np.float32)
            R, t = cam_param['extrinsic'][:,:3,:3].astype(np.float32), cam_param['extrinsic'][:,:3,3].astype(np.float32)
            assert len(R) == len(t)
            assert len(R) == len(img_paths[capture_id])
            for i, frame_idx in enumerate(sorted(list(img_paths[capture_id].keys()))):
                cam_params[capture_id][frame_idx] = {'focal': focal, 'princpt': princpt, 'R': R[i], 't': t[i]}
           
            # make frame index
            for frame_idx in img_paths[capture_id].keys():
                frame_idx_list.append({'capture_id': capture_id, 'frame_idx': frame_idx})

        return img_paths, depthmap_paths, smplx_params, cam_params, frame_idx_list
    
    def load_id_info(self):
        with open(osp.join(self.root_path, 'smplx_optimized', 'shape_param.json')) as f:
            shape_param = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'face_offset.json')) as f:
            face_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'joint_offset.json')) as f:
            joint_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'locator_offset.json')) as f:
            locator_offset = torch.FloatTensor(json.load(f))
        smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

        texture_path = osp.join(self.root_path, 'smplx_optimized', 'face_texture.png')
        texture = torch.FloatTensor(cv2.imread(texture_path)[:,:,::-1].copy().transpose(2,0,1))/255
        texture_mask_path = osp.join(self.root_path, 'smplx_optimized', 'face_texture_mask.png')
        texture_mask = torch.FloatTensor(cv2.imread(texture_mask_path).transpose(2,0,1))/255
        flame.set_texture(texture, texture_mask)

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        capture_id, frame_idx = self.frame_idx_list[idx]['capture_id'], self.frame_idx_list[idx]['frame_idx']

        # load image
        img = load_img(self.img_paths[capture_id][frame_idx])
        img = self.transform(img.astype(np.float32))/255.

        # get mask from depthmap
        depthmap = cv2.imread(self.depthmap_paths[capture_id][frame_idx], -1)[:,:,None]
        mask = (depthmap < depthmap.max()).astype(np.float32) # 0: bkg, 1: human
        y, x = np.where(mask[:,:,0])
        bbox = get_bbox(np.stack((x,y),1), np.ones_like(x))
        mask = self.transform(mask.astype(np.float32))[0,None,:,:]

        data = {'img': img, 'mask': mask, 'bbox': bbox, 'cam_param': self.cam_params[capture_id][frame_idx], 'capture_id': capture_id, 'frame_idx': frame_idx}
        return data
