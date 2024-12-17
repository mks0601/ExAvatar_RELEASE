import cv2
import numpy as np
import torch
import os.path as osp
from glob import glob
from pytorch3d.io import load_ply
import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PerspectiveCameras,
RasterizationSettings,
MeshRasterizer)
import json
from tqdm import tqdm
import argparse

def render_depthmap(mesh, face, cam_param, render_shape):
    mesh = mesh.cuda()[None,:,:]
    face = face.cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}

    batch_size, vertex_num = mesh.shape[:2]
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()

    # render
    with torch.no_grad():
        fragments = rasterizer(mesh)
    
    depthmap = fragments.zbuf.cpu().numpy()[0,:,:,0]
    return depthmap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
out_path = osp.join(root_path, 'depthmaps')
os.makedirs(out_path, exist_ok=True)

# run DepthAnything-V2
assert osp.isfile('./checkpoints/depth_anything_v2_vitl.pth'), 'Please download depth_anything_v2_vitl.pth'
cmd = 'python run.py --encoder vitl --img-path ' + osp.join(root_path, 'frames') + '  --outdir ' + out_path + ' --pred-only --grayscale'
os.system(cmd)

# make background point cloud
depthmap_save = 0
color_save = 0
is_bkg_save = 0
depthmap_path_list = glob(osp.join(out_path, '*.png'))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in depthmap_path_list])
img_height, img_width = cv2.imread(depthmap_path_list[0]).shape[:2]
video_save = cv2.VideoWriter(osp.join(root_path, 'depthmaps.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
for frame_idx in tqdm(frame_idx_list):
    
    # load image
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    img = cv2.imread(img_path).astype(np.float32)

    # load depthmap from DepthAnything-V2
    depthmap_path = osp.join(root_path, 'depthmaps', str(frame_idx) + '.png')
    depthmap = cv2.imread(depthmap_path)
    
    # save video
    frame = np.concatenate((img, depthmap),1)
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame.astype(np.uint8))
    
    # load SMPLX mesh and render depthmap from it
    smplx_mesh_path = osp.join(root_path, 'smplx_optimized', 'meshes_smoothed', str(frame_idx) + '_smplx.ply')
    if not osp.isfile(smplx_mesh_path):
        continue
    smplx_vert, smplx_face = load_ply(smplx_mesh_path)
    with open(osp.join(root_path, 'cam_params', str(frame_idx) + '.json')) as f:
        cam_param = json.load(f)
    smplx_depthmap = render_depthmap(smplx_vert, smplx_face, cam_param, (img_height, img_width))
    smplx_is_fg = smplx_depthmap > 0

    # normalize depthmap from DepthAnything-V2
    depthmap = 255 - depthmap[:,:,0] # close points high values -> close points low values
    scale = np.abs(depthmap[smplx_is_fg] - depthmap[smplx_is_fg].mean()).mean()
    scale_smplx = np.abs(smplx_depthmap[smplx_is_fg] - smplx_depthmap[smplx_is_fg].mean()).mean()
    depthmap = depthmap / scale * scale_smplx
    depthmap = depthmap - depthmap[smplx_is_fg].mean() + smplx_depthmap[smplx_is_fg].mean()

    # load mask
    mask_path = osp.join(root_path, 'masks', str(frame_idx) + '.png')
    mask = cv2.imread(mask_path)[:,:,0]
    is_bkg = mask < 0.5

    # save background points
    depthmap_save += depthmap * is_bkg
    color_save += img * is_bkg[:,:,None]
    is_bkg_save += is_bkg

# save background point cloud
depthmap_save /= (is_bkg_save + 1e-6)
color_save /= (is_bkg_save[:,:,None] + 1e-6)
f = open(osp.join(root_path, 'bkg_point_cloud.txt'), 'w')
for i in tqdm(range(img_height)):
    for j in range(img_width):
        if is_bkg_save[i][j]:
            x = (j - cam_param['princpt'][0]) / cam_param['focal'][0] * depthmap_save[i][j]
            y = (i - cam_param['princpt'][1]) / cam_param['focal'][1] * depthmap_save[i][j]
            z = depthmap_save[i][j]
            rgb = color_save[i][j]
            f.write(str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(rgb[0]) + ' ' + str(rgb[1]) + ' ' + str(rgb[2]) + '\n')
f.close()
