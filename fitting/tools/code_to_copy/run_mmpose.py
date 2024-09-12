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

def load_output(path, frame_idx_list):
    kpts = {}
    with open(path) as f:
        _kpts = json.load(f)
    for i in range(len(_kpts['instance_info'])):
        frame_idx = kpts['instance_info'][i]['frame_id'] - 1 # 1-based -> 0-based
        frame_idx = frame_idx_list[frame_idx]
        xy = np.array(_kpts['instance_info'][i]['instances'][0]['keypoints'], dtype=np.float32).reshape(-1,2)
        score = np.array(_kpts['instance_info'][i]['instances'][0]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
        kpts[frame_idx] = np.concatenate((xy, score),1)
    return kpts

args = parse_args()
root_path = args.root_path

output_root = './vis_results'
os.system('rm -rf ' + output_root)

# run mmpose
input_path = osp.join(root_path, 'video.mp4')
cmd = 'python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py dw-ll_ucoco_384.pth  --input ' + input_path + ' --output-root ' + output_root + ' --save-predictions'
print(cmd)
os.system(cmd)

# results_video.json -> $FRAME_IDX.json
os.makedirs(osp.join(root_path, 'keypoints_whole_body'), exist_ok=True)
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(root_path, 'frames', '*.png'))])
output_path = osp.join(output_root, 'results_video.json')
with open(output_path) as f:
    out = json.load(f)
for i in range(len(out['instance_info'])):
    idx = out['instance_info'][i]['frame_id'] - 1 # 1-based -> 0-based
    frame_idx = frame_idx_list[idx]
    xy = np.array(out['instance_info'][i]['instances'][0]['keypoints'], dtype=np.float32).reshape(-1,2)
    score = np.array(out['instance_info'][i]['instances'][0]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
    kpt = np.concatenate((xy, score),1) # x, y, score
    with open(osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json'), 'w') as f:
        json.dump(kpt.tolist(), f)

# add original image and frame index to the video
output_path = osp.join(output_root, 'video.mp4')
vidcap_orig = cv2.VideoCapture(output_path)
success, frame_vis = vidcap_orig.read()
img_height, img_width = frame_vis.shape[:2]
video_save = cv2.VideoWriter(osp.join(root_path, 'keypoints_whole_body.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height)) 
idx = 0
while success:
    frame_idx = frame_idx_list[idx]
    img = cv2.imread(osp.join(root_path, 'frames', str(frame_idx) + '.png'))
    frame_vis = np.concatenate((img, frame_vis),1)
    frame_vis = cv2.putText(frame_vis, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame_vis)

    success, frame_vis = vidcap_orig.read()
    idx += 1
video_save.release()
