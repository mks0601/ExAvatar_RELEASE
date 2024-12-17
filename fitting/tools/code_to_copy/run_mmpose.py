import os
import os.path as osp
from glob import glob
import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm

# setup: https://mmpose.readthedocs.io/en/latest/installation.html

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

output_root = './vis_results'
os.system('rm -rf ' + output_root)

# run mmpose
assert osp.isfile('./dw-ll_ucoco_384.pth'), 'Please download dw-ll_ucoco_384.pth'
assert osp.isfile('./rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'), 'Please download rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
cmd = 'python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py dw-ll_ucoco_384.pth  --input ' + osp.join(root_path, 'frames') + ' --output-root ' + output_root + ' --save-predictions'
print(cmd)
os.system(cmd)

# move predictions to the root path
os.makedirs(osp.join(root_path, 'keypoints_whole_body'), exist_ok=True)
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(root_path, 'frames', '*.png'))])
output_path_list = glob(osp.join(output_root, '*.json'))
for output_path in output_path_list:
    frame_idx = int(output_path.split('/')[-1].split('results_')[1][:-5])
    with open(output_path) as f:
        out = json.load(f)

    kpt_save = None
    for i in range(len(out['instance_info'])):
        xy = np.array(out['instance_info'][i]['keypoints'], dtype=np.float32).reshape(-1,2)
        score = np.array(out['instance_info'][i]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
        kpt = np.concatenate((xy, score),1) # x, y, score
        if (kpt_save is None) or (kpt_save[:,2].mean() < kpt[:,2].mean()):
            kpt_save = kpt
    with open(osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json'), 'w') as f:
        json.dump(kpt_save.tolist(), f)

# add original image and frame index to the video
output_path_list = glob(osp.join(output_root, '*.png'))
img_height, img_width = cv2.imread(output_path_list[0]).shape[:2]
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(output_root, '*.png'))])
video_save = cv2.VideoWriter(osp.join(root_path, 'keypoints_whole_body.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height)) 
for frame_idx in frame_idx_list:
    img = cv2.imread(osp.join(root_path, 'frames', str(frame_idx) + '.png'))
    output = cv2.imread(osp.join(output_root, str(frame_idx) + '.png'))
    vis = np.concatenate((img, output),1)
    vis = cv2.putText(vis, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(vis)
video_save.release()
