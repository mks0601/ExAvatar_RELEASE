import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import cv2
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from utils.vis import render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    assert args.motion_path, 'Motion path for the animation is required.'
    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    # load ID information
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    tester.smplx_params = None
    tester._make_model()
    
    motion_name = args.motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))])
    render_shape = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx_list[0]) + '.png')).shape[:2]
    video_out = cv2.VideoWriter(motion_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (render_shape[1]*3, render_shape[0]))
    for frame_idx in tqdm(frame_idx_list):
        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}

        # forward
        with torch.no_grad():
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = tester.model.module.human_gaussian(smplx_param, cam_param, is_world_coord=False)
            human_render = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param)
       
        # smplx mesh render
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint_part['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint_part['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint_part['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)
        shape = tester.model.module.human_gaussian.shape_param[None]
        face_offset = smpl_x.face_offset.cuda()[None]
        joint_offset = tester.model.module.human_gaussian.joint_offset[None]
        output = tester.model.module.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        mesh = output.vertices[0]
        mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)

        img = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx) + '.png'))
        render = (human_render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
       
        font_size = 1.5
        thick = 3
        cv2.putText(img, 'image', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
        cv2.putText(mesh_render, 'rendered SMPL-X mesh', (int(1/5*mesh_render.shape[1]), int(0.05*mesh_render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2)
        cv2.putText(render, 'render', (int(1/3*render.shape[1]), int(0.05*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 

        out = np.concatenate((img, mesh_render, render),1).astype(np.uint8)
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.05), int(out.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, 2)
        video_out.write(out)
    
    video_out.release()
    
if __name__ == "__main__":
    main()
