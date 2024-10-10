import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import cv2
import math
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from pytorch3d.renderer import look_at_view_transform
from utils.vis import render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    args = parser.parse_args()

    assert args.subject_id, "Please set sequence name"
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
    
    motion_path = args.motion_path
    if motion_path[-1] == '/':
        motion_name = motion_path[:-1].split('/')[-1]
    else:        
        motion_name = motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))])
    render_shape = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx_list[0]) + '.png')).shape[:2]
    video_out = cv2.VideoWriter(motion_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (render_shape[1]*3, render_shape[0]))
    for i, frame_idx in enumerate(tqdm(frame_idx_list)):
        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}

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
        mesh, root_joint_cam = output.vertices[0], output.joints[0][smpl_x.root_joint_idx]
        mesh = torch.matmul(torch.inverse(cam_param['R']), (mesh - cam_param['t'].view(-1,3)).permute(1,0)).permute(1,0) # camera coordinate -> world coordinate
        root_joint_world = torch.matmul(torch.inverse(cam_param['R']), root_joint_cam - cam_param['t']) # camera coordinate -> world coordinate

        # make camera parmeters with look_at function
        azim = math.pi + math.pi*16*i/len(frame_idx_list) # azim angle of the camera
        if i == 0:
            at_point_orig = root_joint_world.clone()
            at_point = root_joint_world # world coordinate
            cam_pos = torch.matmul(torch.inverse(cam_param['R']), -cam_param['t'].view(3,1)).view(3) # get camera position (world coordinate system)
            at_point_cam = root_joint_cam # camera coordinate
            elev = torch.arctan(torch.abs(at_point_cam[1])/torch.abs(at_point_cam[2])) # elev angle of the camera
            dist = torch.sqrt(torch.sum((cam_pos - at_point)**2)) # distance between camera and mesh
        mesh[:,[0,2]] = mesh[:,[0,2]] - root_joint_world[None,[0,2]] + at_point_orig[None,[0,2]]
        R, t = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=False, at=at_point[None,:], up=((0,1,0),)) 
        R = torch.inverse(R)
        cam_param_rot = {'R': R[0].cuda(), 't': t[0].cuda(), 'focal': cam_param['focal'], 'princpt': cam_param['princpt']}

        mesh = torch.matmul(cam_param_rot['R'], mesh.permute(1,0)).permute(1,0) + cam_param_rot['t'].view(1,3) # world coordinate -> camera coordinate
        mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)

        # forward
        with torch.no_grad():
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = tester.model.module.human_gaussian(smplx_param, cam_param)
            human_asset['mean_3d'][:,[0,2]] = human_asset['mean_3d'][:,[0,2]] - root_joint_world[None,[0,2]] + at_point_orig[None,[0,2]]
            human_render = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param_rot)
       
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
