import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh
import json
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from glob import glob
from tqdm import tqdm

def get_one_box(det_output):
    max_score = 0
    max_bbox = None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = score

    return max_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
root_path = args.root_path

# snapshot load
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
save_path = osp.join(root_path, 'smplx_init')
os.makedirs(save_path, exist_ok=True)
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
video_save = cv2.VideoWriter(osp.join(root_path, 'smplx_init.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
bbox = None
for frame_idx in tqdm(frame_idx_list):
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    det_transform = T.Compose([T.ToTensor()])
    det_input = det_transform(original_img).cuda()
    det_output = det_model([det_input])[0]
    bbox = get_one_box(det_output) # xyxy
    if bbox is None:
        continue
    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] # xywh
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

    # render mesh
    vis_img = original_img[:,:,::-1].copy()
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
    frame = np.concatenate((vis_img, rendered_img),1)
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame.astype(np.uint8))

    # save SMPL-X parameters
    root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
    body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
    lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
    rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
    jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
    shape = out['smplx_shape'].detach().cpu().numpy()[0]
    expr = out['smplx_expr'].detach().cpu().numpy()[0] 
    with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
        json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                'body_pose': body_pose.reshape(-1,3).tolist(), \
                'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                'leye_pose': [0,0,0],\
                'reye_pose': [0,0,0],\
                'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                'shape': shape.reshape(-1).tolist(), \
                'expr': expr.reshape(-1).tolist()}, f)

