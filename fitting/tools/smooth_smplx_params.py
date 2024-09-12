import torch
import numpy as np
import os
import os.path as osp
from scipy.signal import savgol_filter
import json
from glob import glob
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle
from pytorch3d.io import save_ply
import cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
import argparse
import sys
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from utils.smpl_x import smpl_x

def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4

    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result

def smoothen_poses(poses, window_length):
    """Smooth joint angles. Poses and global_root_orient should be given as rotation vectors."""
    n_joints = poses.shape[1] // 3

    # Convert poses to quaternions.
    qs = matrix_to_quaternion(axis_angle_to_matrix(torch.FloatTensor(poses).view(-1,3))).numpy()
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)

    # Smooth the quaternions.
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)

    ps_clean = matrix_to_axis_angle(quaternion_to_matrix(torch.FloatTensor(qs_clean).view(-1,4))).numpy()
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])
    return ps_clean

def render_mesh(mesh, face, cam_param, bkg, blend_ratio=1.0):
    mesh = torch.FloatTensor(mesh).cuda()[None,:,:]
    face = torch.LongTensor(face.astype(np.int64)).cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    materials = Materials(
	device='cuda',
	specular_color=[[0.0, 0.0, 0.0]],
	shininess=0.0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
    
    # background masking
    is_bkg = (fragments.zbuf <= 0).float().cpu().numpy()[0]
    render = images[0,:,:,:3].cpu().numpy()
    fg = render * blend_ratio + bkg/255 * (1 - blend_ratio)
    render = fg * (1 - is_bkg) * 255 + bkg * is_bkg
    return render

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--smooth_length', type=int, default=9, dest='smooth_length')
    args = parser.parse_args()
    return args

# get paths
args = parse_args()
root_path = args.root_path
smplx_param_path_list = glob(osp.join(root_path, 'smplx_optimized', 'smplx_params', '*.json'))
frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in smplx_param_path_list])
frame_num = len(frame_idx_list)

# load smplx parameters of all frames
smplx_params = {}
for frame_idx in frame_idx_list:
    smplx_param_path = osp.join(root_path, 'smplx_optimized', 'smplx_params', str(frame_idx) + '.json')
    with open(smplx_param_path) as f:
        smplx_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
    smplx_params[frame_idx] = smplx_param
    keys = smplx_param.keys()

# smooth smplx parameters
for key in keys:
    if 'pose' in key:
        pose = np.stack([smplx_params[frame_idx][key].reshape(-1) for frame_idx in frame_idx_list])
        pose = smoothen_poses(pose, window_length=args.smooth_length)
        for i, frame_idx in enumerate(frame_idx_list):
            smplx_params[frame_idx][key] = pose[i]
            if key in ['body_pose', 'lhand_pose', 'rhand_pose']:
                smplx_params[frame_idx][key] = smplx_params[frame_idx][key].reshape(-1,3)
    else:
        item = np.stack([smplx_params[frame_idx][key] for frame_idx in frame_idx_list])
        item = savgol_filter(item, window_length=args.smooth_length, polyorder=2, axis=0)
        for i, frame_idx in enumerate(frame_idx_list):
            smplx_params[frame_idx][key] = item[i]

# save smoothed smplx parameters and meshes
param_save_path = osp.join(root_path, 'smplx_optimized', 'smplx_params_smoothed')
mesh_save_path = osp.join(root_path, 'smplx_optimized', 'meshes_smoothed')
render_save_path = osp.join(root_path, 'smplx_optimized', 'renders_smoothed')
os.makedirs(param_save_path, exist_ok=True)
os.makedirs(mesh_save_path, exist_ok=True)
os.makedirs(render_save_path, exist_ok=True)
with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
    shape_param = torch.FloatTensor(json.load(f))
with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
    joint_offset = torch.FloatTensor(json.load(f))
with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
    face_offset = torch.FloatTensor(json.load(f))
img_height, img_width = cv2.imread(osp.join(root_path, 'frames', str(frame_idx_list[0]) + '.png')).shape[:2]
video_save = cv2.VideoWriter(osp.join(root_path, 'smplx_optimized_smoothed.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
for frame_idx in tqdm(frame_idx_list):
    smplx_param = smplx_params[frame_idx]

    # save smplx parameter
    with open(osp.join(param_save_path, str(frame_idx) + '.json'), 'w') as f:
        json.dump({k: v.tolist() for k,v in smplx_param.items()}, f)
    
    # save smplx mesh
    with torch.no_grad():
        output = smpl_x.layer(
                                global_orient = torch.FloatTensor(smplx_param['root_pose']).view(1,-1), \
                                body_pose = torch.FloatTensor(smplx_param['body_pose']).view(1,-1), \
                                jaw_pose = torch.FloatTensor(smplx_param['jaw_pose']).view(1,-1), \
                                leye_pose = torch.FloatTensor(smplx_param['leye_pose']).view(1,-1), \
                                reye_pose = torch.FloatTensor(smplx_param['reye_pose']).view(1,-1), \
                                left_hand_pose = torch.FloatTensor(smplx_param['lhand_pose']).view(1,-1), \
                                right_hand_pose = torch.FloatTensor(smplx_param['rhand_pose']).view(1,-1), \
                                expression = torch.FloatTensor(smplx_param['expr']).view(1,-1), \
                                transl = torch.FloatTensor(smplx_param['trans']).view(1,-1), \
                                betas = shape_param.view(1,-1), \
                                joint_offset = joint_offset.view(-1,3), \
                                face_offset = face_offset.view(-1,3)
                                )
    vert = output.vertices[0]
    save_ply(osp.join(mesh_save_path, str(frame_idx) + '_smplx.ply'), vert, torch.LongTensor(smpl_x.face))
    
    # render smplx mesh
    # load camera parmaeter
    cam_param_path = osp.join(root_path, 'cam_params', str(frame_idx) + '.json')
    with open(cam_param_path) as f:
        cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
    # load image
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    img = cv2.imread(img_path)
    # render
    render = render_mesh(vert.numpy(), smpl_x.face, {'focal': cam_param['focal'], 'princpt': cam_param['princpt']}, img, 1.0)
    # save
    cv2.imwrite(osp.join(render_save_path, str(frame_idx) + '_smplx.jpg'), render)
    frame = np.concatenate((img, render),1)
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame.astype(np.uint8))


