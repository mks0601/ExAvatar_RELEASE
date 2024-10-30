import os
import os.path as osp
import cv2
from glob import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

# path
colmap_path = './colmap_tmp'
os.makedirs(colmap_path, exist_ok=True)
os.system('rm -rf ' + osp.join(colmap_path, '*'))
os.makedirs(osp.join(colmap_path, 'images'), exist_ok=True)
os.system('rm -rf ' + osp.join(colmap_path, 'images', '*'))
os.makedirs(osp.join(colmap_path, 'sparse'), exist_ok=True)
os.system('rm -rf ' + osp.join(colmap_path, 'sparse', '*'))

# prepare inputs of COLMAP (images/imageN.jpg)
with open(osp.join(root_path, 'frame_list_all.txt')) as f:
    frame_idx_list = [int(x) for x in f.readlines()]
for frame_idx in frame_idx_list:
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    img = cv2.imread(img_path)
    cv2.imwrite(osp.join(colmap_path, 'images', 'image' + str(frame_idx) + '.jpg'), img)

# run COLMAP
cmd = 'colmap feature_extractor --database_path ' + osp.join(colmap_path, 'database.db') + ' --image_path ' + osp.join(colmap_path, 'images') + ' --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1'
os.system(cmd)
cmd = 'colmap sequential_matcher --database_path ' + osp.join(colmap_path, 'database.db')
os.system(cmd)
cmd = 'colmap mapper --database_path ' + osp.join(colmap_path, 'database.db') + ' --image_path ' + osp.join(colmap_path, 'images') + ' --output_path ' + osp.join(colmap_path, 'sparse')
os.system(cmd)
cmd = 'colmap model_converter --input_path ' + osp.join(colmap_path, 'sparse', '0') + ' --output_path ' + osp.join(colmap_path, 'sparse', '0') + ' --output_type TXT'
os.system(cmd)

# move outputs of COLMAP to the root path
os.makedirs(osp.join(root_path, 'sparse'), exist_ok=True)
cmd = 'mv ' + osp.join(colmap_path, 'sparse', '0', '*.txt') + ' ' + osp.join(root_path, 'sparse', '.')
os.system(cmd)

# remove temporal directories
cmd = 'rm -rf ' + colmap_path
os.system(cmd)
