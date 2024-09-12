# use original images (without crop and resize) following https://github.com/aipixel/GaussianAvatar/blob/main/eval.py

import cv2
import torch
import os.path as osp
from glob import glob
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--include_bkg', dest='include_bkg', action='store_true')
    args = parser.parse_args()
    assert args.output_path, "Please set output_path."
    assert args.subject_id, "Please set subject_id."
    return args

# get path
args = parse_args()
output_path = args.output_path
subject_id = args.subject_id
include_bkg = args.include_bkg

results = {'psnr': [], 'ssim': [], 'lpips': []}
psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

with open(osp.join('..', 'data', 'NeuMan', 'data', subject_id, 'test_split.txt')) as f:
    lines = f.readlines()
for line in lines:
    frame_idx = int(line[:-5])

    # output image
    out_path = osp.join(output_path, str(frame_idx) + '_scene_human_refined_composed.png')
    out = cv2.imread(out_path)[:,:,::-1]/255.
    out = torch.FloatTensor(out).permute(2,0,1)[None,:,:,:].cuda()
    
    # gt image
    gt_path = osp.join('..', 'data', 'NeuMan', 'data', subject_id, 'images', '%05d.png' % frame_idx)
    gt = cv2.imread(gt_path)[:,:,::-1]/255.
    gt = torch.FloatTensor(gt).permute(2,0,1)[None,:,:,:].cuda()
    
    # gt mask
    mask_path = osp.join('..', 'data', 'NeuMan', 'data', subject_id, 'segmentations', '%05d.png' % frame_idx)
    mask = cv2.imread(mask_path)
    mask = 1 - mask/255. # 0: bkg, 1: human
    mask = torch.FloatTensor(mask).permute(2,0,1)[None,:,:,:].cuda()
    
    # exclude background pixels
    if not include_bkg:
        out = out * mask + 1 * (1 - mask)
        gt = gt * mask + 1 * (1 - mask)

    results['psnr'].append(psnr(out, gt))
    results['ssim'].append(ssim(out, gt))
    results['lpips'].append(lpips(out*2-1, gt*2-1)) # normalize to [-1,1]

print('output path: ' + output_path)
print('subject_id: ' + subject_id)
print('include_bkg: ' + str(include_bkg))
print({k: torch.FloatTensor(v).mean() for k,v in results.items()})

