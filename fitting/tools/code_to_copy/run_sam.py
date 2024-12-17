from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import argparse

def get_bbox(kpt_img, kpt_valid, extend_ratio=1.2):
    x_img, y_img = kpt_img[:,0], kpt_img[:,1]
    x_img = x_img[kpt_valid==1]; y_img = y_img[kpt_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
out_path = osp.join(root_path, 'masks')
os.makedirs(out_path, exist_ok=True)

# load SAM 
ckpt_path = './sam_vit_h_4b8939.pth'
model_type = "vit_h"
assert osp.isfile(ckpt_path), 'Please download sam_vit_h_4b8939.pth'
sam = sam_model_registry[model_type](checkpoint=ckpt_path).cuda()
predictor = SamPredictor(sam)

# run SAM
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
video_save = cv2.VideoWriter(osp.join(root_path, 'masks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
for frame_idx in tqdm(frame_idx_list):
    
    # load image
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    img = cv2.imread(img_path)
    
    # load keypoints
    kpt_path = osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json')
    with open(kpt_path) as f:
        kpt = np.array(json.load(f), dtype=np.float32)
    kpt = kpt[kpt[:,2] > 0.5,:2]
    bbox = get_bbox(kpt, np.ones_like(kpt[:,0]))
    bbox[2:] += bbox[:2] # xywh -> xyxy

    # use keypoints as prompts
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_input)
    masks, scores, logits = predictor.predict(point_coords=kpt, point_labels=np.ones_like(kpt[:,0]), box=bbox[None,:], multimask_output=False)
    mask_input = logits[np.argmax(scores), :, :]
    masks, _, _ = predictor.predict(point_coords=kpt, point_labels=np.ones_like(kpt[:,0]), box=bbox[None,:], multimask_output=False, mask_input=mask_input[None])
    mask = masks.sum(0) > 0

    # save mask and video
    cv2.imwrite(osp.join(out_path, str(frame_idx) + '.png'), mask * 255)
    img_masked = img.copy()
    img_masked[~mask] = 0
    frame = np.concatenate((img, img_masked),1)
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame.astype(np.uint8))

