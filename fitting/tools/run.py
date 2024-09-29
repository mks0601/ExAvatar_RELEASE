import os
import os.path as osp
from glob import glob
import sys
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
if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]

# remove unnecessary frames
with open(osp.join(root_path, 'frame_list_all.txt')) as f:
    frame_idx_list = [int(x) for x in f.readlines()]
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
for img_path in img_path_list:
    frame_idx = int(img_path.split('/')[-1][:-4])
    if frame_idx not in frame_idx_list:
        cmd = 'rm ' + img_path
        result = os.system(cmd)
        if (result != 0):
            print('something bad happened when removing unnecessary frames. terminate the script.')
            sys.exit()

# make camera parameters
cmd = 'python make_virtual_cam_params.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when making the virtual camera parameters. terminate the script.')
    sys.exit()

# DECA (get initial FLAME parameters)
os.chdir('./DECA')
cmd = 'python run_deca.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running DECA. terminate the script.')
    sys.exit()
os.chdir('..')

# Hand4Whole (get initial SMPLX parameters)
os.chdir('./Hand4Whole_RELEASE/demo')
cmd = 'python run_hand4whole.py --gpu 0 --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running Hand4Whole. terminate the script.')
    sys.exit()
os.chdir('../../')

# mmpose (get 2D whole-body keypoints)
os.chdir('./mmpose')
cmd = 'python run_mmpose.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running mmpose. terminate the script.')
    sys.exit()
os.chdir('..')

# fit SMPLX
os.chdir('../main')
cmd = 'python fit.py --subject_id ' + subject_id
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting. terminate the script.')
    sys.exit()
os.chdir('../tools')
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, '*') + ' ' + osp.join(root_path, '.')
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when moving the fitted files to root_path. terminate the script.')
    sys.exit()

# unwrap textures of FLAME
os.chdir('../main')
cmd = 'python unwrap.py --subject_id ' + subject_id
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when unwrapping the face images to FLAME UV texture. terminate the script.')
    sys.exit()
os.chdir('../tools')
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, 'unwrapped_textures', '*') + ' ' + osp.join(root_path, 'smplx_optimized', '.')
result = os.system(cmd)
if (result != 0):
    print('something bad happened when moving the unwrapped FLAME UV texture to root_path. terminate the script.')
    sys.exit()

# smooth SMPLX
cmd = 'python smooth_smplx_params.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when smoothing smplx parameters. terminate the script.')
    sys.exit()

