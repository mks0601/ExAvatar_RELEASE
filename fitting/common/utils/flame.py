import numpy as np
import torch
import os.path as osp
from config import cfg
import smplx
from nets.layer import get_face_index_map_uv
import pickle

class FLAME(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.layer_arg = {'create_betas': False, 'create_expression': False, 'create_global_orient': False, 'create_neck_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_transl': False}
        self.layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_face_contour=True, **self.layer_arg)
        self.vertex_num = 5023
        self.face = self.layer.faces.astype(np.int64)
        self.vertex_uv, self.face_uv = self.load_texture_model()

        # joint
        self.joint = {
                'num': 5,
                'name': ('Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye'),
                'root_idx': 0
                }

        # keypoint
        self.kpt = {
                'num': 75,
                'name': ['Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye'] +  ['Face_' + str(i) for i in range(1,69)] + ['L_Ear', 'R_Ear'],
                'root_idx': 0
                }

        # vertex idxs
        self.lear_vertex_idx = 160
        self.rear_vertex_idx = 1167
        
        # UV mask
        self.uv_mask = self.make_uv_mask()
   
    def load_texture_model(self):
        texture = dict(np.load(osp.join(cfg.human_model_path, 'flame', 'FLAME_texture.npz'), allow_pickle=True, encoding='latin1'))
        vertex_uv, face_uv = texture['vt'], texture['ft'].astype(np.int64)
        vertex_uv[:,1] = 1 - vertex_uv[:,1]
        return vertex_uv, face_uv
    
    def make_uv_mask(self):
        vertex_uv = torch.from_numpy(self.vertex_uv).float().cuda()[None,:,:]
        face_uv = torch.from_numpy(self.face_uv).long().cuda()[None,:,:]
        outputs = get_face_index_map_uv(vertex_uv, face_uv, cfg.uvmap_shape)
        pix_to_face = outputs.pix_to_face # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel. invalid: -1
        uv_mask = pix_to_face[0,:,:,0]
        
        # make neck region invalid
        is_neck = torch.zeros((self.vertex_num)).float()
        is_neck[self.layer.lbs_weights.argmax(1)==self.joint['root_idx']] = 1
        for i in range(len(self.face)):
            v0, v1, v2 = self.face[i]
            if is_neck[v0] or is_neck[v1] or is_neck[v2]:
                uv_mask[uv_mask==i] = -1

        # make region without facial expression invalid
        with open(osp.join(cfg.human_model_path, 'flame', '2019', 'generic_model.pkl'), 'rb') as f:
            flame_2019 = pickle.load(f, encoding='latin1')
        expr_vertex_idx = np.where((flame_2019['shapedirs'][:,:,300:300+self.expr_param_dim] != 0).sum((1,2)) > 0)[0] # FLAME.SHAPE_SPACE_DIM == 300
        expr_vertex_mask = torch.zeros((self.vertex_num)).float()
        expr_vertex_mask[expr_vertex_idx] = 1
        for i in range(len(self.face)):
            v0, v1, v2 = self.face[i]
            if (not expr_vertex_mask[v0]) or (not expr_vertex_mask[v1]) or (not expr_vertex_mask[v2]):
                uv_mask[uv_mask==i] = -1

        uv_mask = (uv_mask != -1).float()
        return uv_mask
    
flame = FLAME()

