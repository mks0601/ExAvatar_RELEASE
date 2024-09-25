import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
from config import cfg
import smplx
from config import cfg

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


