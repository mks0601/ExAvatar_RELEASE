import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
from config import cfg
import smplx
from config import cfg

from pytorch3d.structures import Meshes
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras, RasterizationSettings, MeshRasterizer
from config import cfg

def get_face_index_map_uv(vertex_uv, face_uv, uvmap_shape):
    # scale UV coordinates to uvmap_shape
    vertex_uv = torch.stack((vertex_uv[:,:,0] * uvmap_shape[1], vertex_uv[:,:,1] * uvmap_shape[0]),2)
    vertex_uv = torch.cat((vertex_uv, torch.ones_like(vertex_uv[:,:,:1])),2) # add dummy depth
    vertex_uv = torch.stack((-vertex_uv[:,:,0], -vertex_uv[:,:,1], vertex_uv[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vertex_uv, face_uv)

    cameras = OrthographicCameras(
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(uvmap_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=uvmap_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    outputs = rasterizer(mesh)
    return outputs

class FLAME(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.layer_arg = {'create_betas': False, 'create_expression': False, 'create_global_orient': False, 'create_neck_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_transl': False}
        self.layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_face_contour=True, **self.layer_arg)
        self.vertex_num = 5023
        self.face = self.layer.faces.astype(np.int64)
        self.vertex_uv, self.face_uv = self.load_texture_model()

    def load_texture_model(self):
        texture = np.load(osp.join(cfg.human_model_path, 'flame', 'FLAME_texture.npz'))
        vertex_uv, face_uv = texture['vt'], texture['ft'].astype(np.int64)
        vertex_uv[:,1] = 1 - vertex_uv[:,1]
        return vertex_uv, face_uv
   
    def set_texture(self, texture, texture_mask):
        self.texture = texture.cuda()
        self.texture_mask = texture_mask.cuda() 


flame = FLAME()


