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
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
for img_path in img_path_list:
    frame_idx = int(img_path.split('/')[-1][:-4])
    img = cv2.imread(img_path)
    cv2.imwrite(osp.join(colmap_path, 'images', 'image' + str(frame_idx) + '.jpg'), img)

# run COLMAP
cmd = 'colmap feature_extractor --database_path ' + osp.join(colmap_path, 'database.db') + ' --image_path ' + osp.join(colmap_path, 'images')
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
