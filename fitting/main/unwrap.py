import argparse
import numpy as np
import cv2
from config import cfg
import torch
import torch.nn as nn
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from utils.flame import flame
from glob import glob
from tqdm import tqdm
from base import Trainer
from pytorch3d.io import load_ply

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, True)
    
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    model = trainer.model.module
    
    face_texture_save, face_texture_mask_save = 0, 0
    for itr, data in enumerate(tqdm(trainer.batch_generator)):
        for k in data.keys():
            if torch.is_tensor(data[k]):
                data[k] = data[k].cuda()
            elif isinstance(data[k], dict):
                for kk in data[k].keys():
                    data[k][kk] = data[k][kk].cuda()
        batch_size = data['img'].shape[0]

        flame_mesh_cam_list = []
        smplx_inputs = {'shape': [], 'expr': [], 'trans': [], 'joint_offset': [], 'locator_offset': []}
        for i in range(batch_size):
            frame_idx = int(data['frame_idx'][i])
            root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id, 'smplx_optimized')
            flame_mesh_cam, _ = load_ply(osp.join(root_path, 'meshes', str(frame_idx) + '_flame.ply'))
            flame_mesh_cam_list.append(flame_mesh_cam)
            with open(osp.join(root_path, 'shape_param.json')) as f:
                smplx_inputs['shape'].append(torch.FloatTensor(json.load(f)))
            with open(osp.join(root_path, 'joint_offset.json')) as f:
                smplx_inputs['joint_offset'].append(torch.FloatTensor(json.load(f)))
            with open(osp.join(root_path, 'locator_offset.json')) as f:
                smplx_inputs['locator_offset'].append(torch.FloatTensor(json.load(f)))
            with open(osp.join(root_path, 'smplx_params', str(frame_idx) + '.json')) as f:
                smplx_param = json.load(f)
            smplx_inputs['expr'].append(torch.FloatTensor(smplx_param['expr']))
            smplx_inputs['trans'].append(torch.FloatTensor(smplx_param['trans']))
        flame_mesh_cam = torch.stack(flame_mesh_cam_list).cuda()
        smplx_inputs = {k: torch.stack(v).cuda() for k,v in smplx_inputs.items()}

        # get coordinates from the initial parameters
        data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
        data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
        data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
        data['smplx_param']['joint_offset'] = smplx_inputs['joint_offset'].clone().detach()
        data['smplx_param']['locator_offset'] = smplx_inputs['locator_offset'].clone().detach()
        smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = model.get_smplx_coord(data['smplx_param'], data['cam_param_proj'], use_face_offset=False)

        # check face visibility
        face_valid = model.check_face_visibility(smplx_mesh_cam_init[:,smpl_x.face_vertex_idx,:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('L_Eye'),:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('R_Eye'),:])
        face_valid = face_valid * data['flame_valid']

        # face unwrap to uv space
        face_texture, face_texture_mask = model.xy2uv(data['img_face'], flame_mesh_cam, flame.face, data['cam_param_face'])
        face_texture = face_texture * flame.uv_mask[None,None,:,:] * face_valid[:,None,None,None]
        face_texture_mask = face_texture_mask * flame.uv_mask[None,None,:,:] * face_valid[:,None,None,None]

        # face unwrapped texture
        face_texture_save += face_texture.sum(0).detach().cpu().numpy()
        face_texture_mask_save += face_texture_mask.sum(0).detach().cpu().numpy()

    # face unwrapped texture
    save_root_path = osp.join(cfg.result_dir, 'unwrapped_textures')
    os.makedirs(save_root_path, exist_ok=True)
    face_texture = face_texture_save / (face_texture_mask_save + 1e-4)
    face_texture_mask = (face_texture_mask_save > 0).astype(np.uint8)
    cv2.imwrite(osp.join(save_root_path, 'face_texture.png'), face_texture.transpose(1,2,0)[:,:,::-1]*255)
    cv2.imwrite(osp.join(save_root_path, 'face_texture_mask.png'), face_texture_mask.transpose(1,2,0)[:,:,::-1]*255)

if __name__ == "__main__":
    main()
