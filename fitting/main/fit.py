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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from base import Trainer
from utils.vis import render_mesh
from pytorch3d.io import save_ply

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def rotation_6d_to_axis_angle(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    # register initial flame parameters
    flame_params = {}
    for frame_idx in trainer.flame_params.keys():
        flame_params[frame_idx] = {}
        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
            flame_params[frame_idx][key] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(trainer.flame_params[frame_idx][key].cuda())))
        flame_params[frame_idx]['expr'] = nn.Parameter(trainer.flame_params[frame_idx]['expr'].cuda())
        flame_params[frame_idx]['trans'] = nn.Parameter(trainer.flame_params[frame_idx]['trans'].cuda())
    flame_shape = nn.Parameter(trainer.flame_shape_param.float().cuda())

    # register initial smplx parameters
    smplx_params = {}
    for frame_idx in trainer.smplx_params.keys():
        smplx_params[frame_idx] = {}
        for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose']:
            smplx_params[frame_idx][key] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(trainer.smplx_params[frame_idx][key].cuda())))
        smplx_params[frame_idx]['jaw_pose'] = flame_params[frame_idx]['jaw_pose'] # share
        smplx_params[frame_idx]['leye_pose'] = flame_params[frame_idx]['leye_pose'] # share
        smplx_params[frame_idx]['reye_pose'] = flame_params[frame_idx]['reye_pose'] # share
        smplx_params[frame_idx]['expr'] = flame_params[frame_idx]['expr'] # share
        smplx_params[frame_idx]['trans'] = nn.Parameter(trainer.smplx_params[frame_idx]['trans'].cuda()) 
    smplx_shape = nn.Parameter(torch.zeros((smpl_x.shape_param_dim)).float().cuda())
    face_offset = nn.Parameter(torch.zeros((flame.vertex_num,3)).float().cuda())
    joint_offset = nn.Parameter(torch.zeros((smpl_x.joint['num'],3)).float().cuda())
    locator_offset = nn.Parameter(torch.zeros((smpl_x.joint['num'],3)).float().cuda())
    
    for epoch in range(cfg.end_epoch):
        cfg.set_itr_opt_num(epoch)
        face_texture_save, face_texture_mask_save = 0, 0

        for itr_data, data in enumerate(trainer.batch_generator):
            batch_size = data['img_orig'].shape[0]

            for itr_opt in range(cfg.itr_opt_num):
                cfg.set_stage(epoch, itr_opt)

                # optimizer
                if (epoch == 0) and (itr_opt == 0):
                    # smplx and flame root pose and translatioin
                    optimizable_params = []
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'trans']:
                            optimizable_params.append(smplx_params[int(frame_idx)][key])
                        for key in ['root_pose', 'trans']:
                            optimizable_params.append(flame_params[int(frame_idx)][key])
                    trainer.get_optimizer(optimizable_params)
                elif ((epoch == 0) and (itr_opt == cfg.stage_itr[0])) or ((epoch > 0) and (itr_opt == 0)):
                    # all parameters
                    if epoch == (cfg.end_epoch - 1):
                        optimizable_params = [] # do not optimize shared parameters to make per-frame parameters consistent with the shared ones
                    else:
                        optimizable_params = [smplx_shape, flame_shape, face_offset, joint_offset, locator_offset] # all shared parameters
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose', 'trans']: # jaw_pose, leye_pose, reye_pose, and expr is provided by flame_params
                            optimizable_params.append(smplx_params[int(frame_idx)][key])
                        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'trans']:
                            optimizable_params.append(flame_params[int(frame_idx)][key])
                    trainer.get_optimizer(optimizable_params)

                # inputs
                smplx_inputs = {'shape': [smplx_shape for _ in range(batch_size)], 'face_offset': [face_offset for _ in range(batch_size)], 'joint_offset': [joint_offset for _ in range(batch_size)], 'locator_offset': [locator_offset for _ in range(batch_size)]}
                flame_inputs = {'shape': [flame_shape for _ in range(batch_size)]}
                for frame_idx in data['frame_idx']:
                    for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                        if key not in smplx_inputs:
                            smplx_inputs[key] = [smplx_params[int(frame_idx)][key]]
                        else:
                            smplx_inputs[key].append(smplx_params[int(frame_idx)][key])
                    for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'trans']:
                        if key not in flame_inputs:
                            flame_inputs[key] = [flame_params[int(frame_idx)][key]]
                        else:
                            flame_inputs[key].append(flame_params[int(frame_idx)][key])
                for key in smplx_inputs.keys():
                    smplx_inputs[key] = torch.stack(smplx_inputs[key])
                for key in flame_inputs.keys():
                    flame_inputs[key] = torch.stack(flame_inputs[key])

                # forwrad
                trainer.set_lr(itr_opt)
                trainer.optimizer.zero_grad()
                loss, out = trainer.model(smplx_inputs, flame_inputs, data, return_output=((epoch==cfg.end_epoch-1) and (itr_opt==cfg.itr_opt_num-1)))
                loss = {k:loss[k].mean() for k in loss}
                
                # backward
                sum(loss[k] for k in loss).backward()
                trainer.optimizer.step()
                print(cfg.result_dir)

                # log
                screen = [
                    'epoch %d/%d itr_data %d/%d itr_opt %d/%d:' % (epoch, cfg.end_epoch, itr_data, trainer.itr_per_epoch, itr_opt, cfg.itr_opt_num),
                    'lr: %g' % (trainer.get_lr()),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                print(screen)

            # save
            if epoch != (cfg.end_epoch-1):
                continue
            save_root_path = osp.join(cfg.result_dir, cfg.subject_id, 'smplx_optimized')
            os.makedirs(save_root_path, exist_ok=True)
            smplx_mesh_cam = out['smplx_mesh_cam'].detach().cpu()
            smplx_mesh_cam_wo_jo = out['smplx_mesh_cam_wo_jo'].detach().cpu()
            smplx_mesh_cam_wo_fo = out['smplx_mesh_cam_wo_fo'].detach().cpu()
            smplx_trans = out['smplx_trans'].detach().cpu().numpy()
            flame_mesh_cam = out['flame_mesh_cam'].detach().cpu()
            smplx_mesh_wo_pose_wo_expr = out['smplx_mesh_wo_pose_wo_expr'].detach().cpu()
            smplx_mesh_wo_pose_wo_expr_wo_fo = out['smplx_mesh_wo_pose_wo_expr_wo_fo'].detach().cpu()
            flame_mesh_wo_pose_wo_expr = out['flame_mesh_wo_pose_wo_expr'].detach().cpu()
            face_texture = out['face_texture'].detach().cpu().numpy()
            face_texture_mask = out['face_texture_mask'].detach().cpu().numpy()
            for i in range(batch_size):
                frame_idx = int(data['frame_idx'][i])

                # mesh
                save_ply(osp.join(save_root_path, 'smplx_wo_pose_wo_expr.ply'), torch.FloatTensor(smplx_mesh_wo_pose_wo_expr[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                save_ply(osp.join(save_root_path, 'smplx_wo_pose_wo_expr_wo_fo.ply'), torch.FloatTensor(smplx_mesh_wo_pose_wo_expr_wo_fo[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                save_ply(osp.join(save_root_path, 'flame_wo_pose_wo_expr.ply'), torch.FloatTensor(flame_mesh_wo_pose_wo_expr[i]).contiguous(), torch.IntTensor(flame.face).contiguous())
                save_path = osp.join(save_root_path, 'meshes')
                os.makedirs(save_path, exist_ok=True)
                save_ply(osp.join(save_path, str(frame_idx) + '_smplx.ply'), torch.FloatTensor(smplx_mesh_cam[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                save_ply(osp.join(save_path, str(frame_idx) + '_smplx_wo_jo.ply'), torch.FloatTensor(smplx_mesh_cam_wo_jo[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                save_ply(osp.join(save_path, str(frame_idx) + '_smplx_wo_fo.ply'), torch.FloatTensor(smplx_mesh_cam_wo_fo[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                save_ply(osp.join(save_path, str(frame_idx) + '_flame.ply'), torch.FloatTensor(flame_mesh_cam[i]).contiguous(), torch.IntTensor(flame.face).contiguous())

                # render
                save_path = osp.join(save_root_path, 'renders')
                os.makedirs(save_path, exist_ok=True)
                render_smplx = render_mesh(smplx_mesh_cam[i].numpy(), smpl_x.face, {'focal': data['cam_param']['focal'][i].numpy(), 'princpt': data['cam_param']['princpt'][i].numpy()}, data['img_orig'][i].numpy()[:,:,::-1], 1.0)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_smplx.jpg'), render_smplx)
                render_smplx_wo_jo = render_mesh(smplx_mesh_cam_wo_jo[i].numpy(), smpl_x.face, {'focal': data['cam_param']['focal'][i].numpy(), 'princpt': data['cam_param']['princpt'][i].numpy()}, data['img_orig'][i].numpy()[:,:,::-1], 1.0)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_smplx_wo_jo.jpg'), render_smplx_wo_jo)
                render_smplx_wo_fo = render_mesh(smplx_mesh_cam_wo_fo[i].numpy(), smpl_x.face, {'focal': data['cam_param']['focal'][i].numpy(), 'princpt': data['cam_param']['princpt'][i].numpy()}, data['img_orig'][i].numpy()[:,:,::-1], 1.0)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_smplx_wo_fo.jpg'), render_smplx_wo_fo)
                render_flame = render_mesh(flame_mesh_cam[i].numpy(), flame.face, {'focal': data['cam_param']['focal'][i].numpy(), 'princpt': data['cam_param']['princpt'][i].numpy()}, data['img_orig'][i].numpy()[:,:,::-1], 1.0)
                cv2.imwrite(osp.join(save_path, str(frame_idx) + '_flame.jpg'), render_flame)

                # smplx parameter
                save_path = osp.join(save_root_path, 'smplx_params')
                os.makedirs(save_path, exist_ok=True)
                with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
                    json.dump({'root_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['root_pose'].detach().cpu()).numpy().tolist(), \
                            'body_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['body_pose'].detach().cpu()).numpy().tolist(), \
                            'jaw_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['jaw_pose'].detach().cpu()).numpy().tolist(), \
                            'leye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['leye_pose'].detach().cpu()).numpy().tolist(), \
                            'reye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['reye_pose'].detach().cpu()).numpy().tolist(), \
                            'lhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['lhand_pose'].detach().cpu()).numpy().tolist(), \
                            'rhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['rhand_pose'].detach().cpu()).numpy().tolist(), \
                            'expr': smplx_params[frame_idx]['expr'].detach().cpu().numpy().tolist(), \
                            'trans': smplx_trans[i].tolist()}, f)
                # shape parameter
                with open(osp.join(save_root_path, 'shape_param.json'), 'w') as f:
                    json.dump(smplx_shape.detach().cpu().numpy().tolist(), f)
                # face offset
                _face_offset = smpl_x.get_face_offset(face_offset[None,:,:])[0]
                with open(osp.join(save_root_path, 'face_offset.json'), 'w') as f:
                    json.dump(_face_offset.detach().cpu().numpy().tolist(), f)
                # joint offset
                _joint_offset = smpl_x.get_joint_offset(joint_offset[None,:,:])[0]
                with open(osp.join(save_root_path, 'joint_offset.json'), 'w') as f:
                    json.dump(_joint_offset.detach().cpu().numpy().tolist(), f)
                # locaotr offset
                _locator_offset = smpl_x.get_locator_offset(locator_offset[None,:,:])[0]
                with open(osp.join(save_root_path, 'locator_offset.json'), 'w') as f:
                    json.dump(_locator_offset.detach().cpu().numpy().tolist(), f)

                # face unwrapped texture
                face_texture_save += face_texture[i]
                face_texture_mask_save += face_texture_mask[i]

        # face unwrapped texture
        if epoch != (cfg.end_epoch-1):
            continue
        face_texture = face_texture_save / (face_texture_mask_save + 1e-4)
        face_texture_mask = (face_texture_mask_save > 0).astype(np.uint8)
        cv2.imwrite(osp.join(save_root_path, 'face_texture.png'), face_texture.transpose(1,2,0)[:,:,::-1]*255)
        cv2.imwrite(osp.join(save_root_path, 'face_texture_mask.png'), face_texture_mask.transpose(1,2,0)[:,:,::-1]*255)
    
    # video
    save_path = osp.join(save_root_path, '..', 'smplx_optimized.mp4')
    video_shape = cv2.imread(glob(osp.join(save_root_path, 'renders', '*.jpg'))[0]).shape[:2] # height, width
    video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_shape[1]*2, video_shape[0]))
    frame_idx_list = sorted([int(x.split('/')[-1].split('_')[0]) for x in glob(osp.join(save_root_path, 'renders', '*_smplx.jpg'))])
    for frame_idx in frame_idx_list:
        orig_img_path = trainer.trainset_loader.img_paths[frame_idx]
        orig_img = cv2.imread(orig_img_path)
        render = cv2.imread(osp.join(save_root_path, 'renders', str(frame_idx) + '_smplx.jpg'))
        out = np.concatenate((orig_img, render),1)
        cv2.putText(out, str(frame_idx), (int(0.02*render.shape[1]), int(0.1*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2.5, [51,51,255], 3, 2) # write frame index
        video_out.write(out)
    video_out.release()



if __name__ == "__main__":
    main()
