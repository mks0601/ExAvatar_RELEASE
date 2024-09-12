import os
import os.path as osp
from glob import glob
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
os.makedirs(osp.join(root_path, 'frames'), exist_ok=True)

vidcap = cv2.VideoCapture(osp.join(root_path, 'video.mp4'))
frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, frame = vidcap.read()
frame_idx = 0
while success:
    print(str(frame_idx) + '/' + str(frame_num), end='\r')
    cv2.imwrite(osp.join(root_path, 'frames', str(frame_idx) + '.png'), frame)
    success, frame = vidcap.read()
    frame_idx += 1
