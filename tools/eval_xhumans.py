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
    args = parser.parse_args()
    assert args.output_path, "Please set output_path."
    assert args.subject_id, "Please set subject_id."
    return args

# get path
args = parse_args()
output_path = args.output_path
subject_id = args.subject_id

# metrics
results = {'psnr': [], 'ssim': [], 'lpips': []}
psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

# measure metrics
capture_path_list = glob(osp.join('..', 'data', 'XHumans', 'data', subject_id, 'test', '*'))
for capture_path in capture_path_list:
    capture_id = capture_path.split('/')[-1]
    frame_idxs = sorted([int(x.split('/')[-1][6:-4]) for x in glob(osp.join(capture_path, 'render', 'image', '*.png'))])
    
    for frame_idx in frame_idxs:
        out_path = osp.join(output_path, subject_id, 'test', capture_id, str(frame_idx) + '_human_refined.png')
        out = cv2.imread(out_path)[:,:,::-1]/255.
        out = torch.FloatTensor(out).permute(2,0,1)[None,:,:,:].cuda()

        gt_path = osp.join(capture_path, 'render', 'image', 'color_%06d.png' % frame_idx)
        gt = cv2.imread(gt_path)[:,:,::-1]/255.
        gt = torch.FloatTensor(gt).permute(2,0,1)[None,:,:,:].cuda()
 
        depthmap_path = osp.join(capture_path, 'render', 'depth', 'depth_%06d.tiff' % frame_idx)
        depthmap = cv2.imread(depthmap_path, -1)
        mask = depthmap < depthmap.max()
        mask = torch.FloatTensor(mask)[None,None,:,:].cuda()

        out = out * mask + 1 * (1 - mask)
        gt = gt * mask + 1 * (1 - mask)

        results['psnr'].append(psnr(out, gt))
        results['ssim'].append(ssim(out, gt))
        results['lpips'].append(lpips(out*2-1, gt*2-1))
 
print('output path: ' + output_path)
print('subject_id: ' + subject_id)
print({k: torch.FloatTensor(v).mean() for k,v in results.items()})
   
