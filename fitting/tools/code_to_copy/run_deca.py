import os
import os.path as osp
import json
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

# run DECA
output_save_path = './flame_parmas_out'
os.system('rm -rf ' + output_save_path)
os.makedirs(output_save_path, exist_ok=True)
cmd = 'python demos/demo_reconstruct.py -i ' + osp.join(root_path, 'frames') + ' --saveDepth True --saveObj True --rasterizer_type=pytorch3d --savefolder ' + output_save_path
os.system(cmd)

# folders -> $FRAME_IDX.json
save_path = osp.join(root_path, 'flame_init', 'flame_params')
os.makedirs(save_path, exist_ok=True)
flame_shape_param = []
output_path_list = [x for x in glob(osp.join(output_save_path, '*.json'))]
for output_path in output_path_list:
    frame_idx = int(output_path.split('/')[-1][:-5])
    with open(output_path) as f:
        flame_param = json.load(f)
    if flame_param['is_valid']:
        root_pose, jaw_pose = torch.FloatTensor(flame_param['pose'])[:,:3].view(3), torch.FloatTensor(flame_param['pose'])[:,3:].view(3)
        shape = torch.FloatTensor(flame_param['shape']).view(-1)
        expr = torch.FloatTensor(flame_param['exp']).view(-1)
        flame_shape_param.append(shape)

        root_pose, jaw_pose, shape, expr = root_pose.tolist(), jaw_pose.tolist(), shape.tolist(), expr.tolist()
        neck_pose, leye_pose, reye_pose = [0,0,0], [0,0,0], [0,0,0]
    else:
         root_pose, jaw_pose, neck_pose, leye_pose, reye_pose, expr, shape = None, None, None, None, None, None, None
    flame_param = {'root_pose': root_pose, 'neck_pose': neck_pose, 'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose, 'expr': expr, 'is_valid': flame_param['is_valid']}
    with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
        json.dump(flame_param, f)
flame_shape_param = torch.stack(flame_shape_param).mean(0).tolist()
with open(osp.join(root_path, 'flame_init', 'shape_param.json'), 'w') as f:
    json.dump(flame_shape_param, f)

# visualization
save_path = osp.join(root_path, 'flame_init', 'renders')
os.makedirs(save_path, exist_ok=True)
vis_path_list = glob(osp.join(output_save_path, '*_vis.jpg'))
for vis_path in vis_path_list:
    frame_idx = int(vis_path.split('/')[-1].split('_')[0])
    cmd = 'cp ' + vis_path + ' ' + osp.join(save_path, str(frame_idx) + '.jpg')
    os.system(cmd)


