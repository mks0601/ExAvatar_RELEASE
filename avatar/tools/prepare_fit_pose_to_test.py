import os
import os.path as osp
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path

# set epoch to 0
src_path = osp.join(root_path, 'snapshot_4.pth')
model = torch.load(src_path)
model['epoch'] = 0

# save
if root_path[-1] == '/':
    tgt_path = root_path[:-1] + '_fit_pose_to_test'
else:
    tgt_path = root_path + '_fit_pose_to_test'
os.makedirs(tgt_path, exist_ok=True) # e.g., ../output/model_dump/bike_fit_pose_to_test
tgt_path = osp.join(tgt_path, 'snapshot_0.pth')
torch.save(model, tgt_path)
