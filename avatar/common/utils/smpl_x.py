import sys
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import cfg
from utils.smplx import smplx
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import math

class SMPLX(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = {gender: smplx.create(cfg.human_model_path, 'smplx', gender=gender, num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg) for gender in ['neutral', 'male', 'female']}
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.layer = {gender: self.get_expr_from_flame(self.layer[gender]) for gender in ['neutral', 'male', 'female']}
        self.vertex_num = 10475
        self.face_orig = self.layer['neutral'].faces.astype(np.int64)
        self.is_cavity, self.face = self.add_cavity()
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.rhand_vertex_idx = hand_vertex_idx['right_hand']
        self.lhand_vertex_idx = hand_vertex_idx['left_hand']
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        # SMPLX joint set
        self.joint_num = 55 # 22 (body joints) + 3 (face joints) + 30 (hand joints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'Jaw', 'L_Eye', 'R_Eye', # face joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
        )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('R_Wrist')+1),
        'face': range(self.joints_name.index('Jaw'), self.joints_name.index('R_Eye')+1),
        'lhand': range(self.joints_name.index('L_Index_1'), self.joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.joints_name.index('R_Index_1'), self.joints_name.index('R_Thumb_3')+1)}
        self.neutral_body_pose = torch.zeros((len(self.joint_part['body'])-1,3)) # 大 pose in axis-angle representation (body pose without root joint)
        self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
        self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        self.neutral_jaw_pose = torch.FloatTensor([1/3, 0, 0])
        
        # subdivider
        self.subdivider_list = self.get_subdivider(2)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled)+1)
 
    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim)
        smplx_layer.expr_dirs[self.face_vertex_idx,:,:] = flame_layer.expr_dirs
        return smplx_layer
       
    def set_id_info(self, shape_param, face_offset, joint_offset, locator_offset):
        self.shape_param = shape_param
        self.face_offset = face_offset
        self.joint_offset = joint_offset
        self.locator_offset = locator_offset

    def get_joint_offset(self, joint_offset):
        weight = torch.ones((1,self.joint_num,1)).float().cuda()
        weight[:,self.root_joint_idx,:] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_subdivider(self, subdivide_num):
        vert = self.layer['neutral'].v_template.float().cuda()
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        subdivider_list = [SubdivideMeshes(mesh)]
        for i in range(subdivide_num-1):
            mesh = subdivider_list[-1](mesh)
            subdivider_list.append(SubdivideMeshes(mesh))
        return subdivider_list

    def upsample_mesh(self, vert, feat_list=None):
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        if feat_list is None:
            for subdivider in self.subdivider_list:
                mesh = subdivider(mesh)
            vert = mesh.verts_list()[0]
            return vert
        else:
            feat_dims = [x.shape[1] for x in feat_list]
            feats = torch.cat(feat_list,1)
            for subdivider in self.subdivider_list:
                mesh, feats = subdivider(mesh, feats)
            vert = mesh.verts_list()[0]
            feats = feats[0]
            feat_list = torch.split(feats, feat_dims, dim=1)
            return vert, *feat_list

    def add_cavity(self):
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        is_cavity = np.zeros((self.vertex_num), dtype=np.float32)
        is_cavity[lip_vertex_idx] = 1.0

        cavity_face = [[0,1,7], [1,2,7], [2, 3,5], [3,4,5], [2,5,6], [2,6,7]]
        face_new = list(self.face_orig)
        for face in cavity_face:
            v1, v2, v3 = face
            face_new.append([lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]])
        face_new = np.array(face_new, dtype=np.int64)
        return is_cavity, face_new
 
    def get_expr_vertex_idx(self):
        # FLAME 2020 has all vertices of expr_vertex_idx. use FLAME 2019
        with open(osp.join(cfg.human_model_path, 'flame', '2019', 'generic_model.pkl'), 'rb') as f:
            flame_2019 = pickle.load(f, encoding='latin1')
        vertex_idxs = np.where((flame_2019['shapedirs'][:,:,300:300+self.expr_param_dim] != 0).sum((1,2)) > 0)[0] # FLAME.SHAPE_SPACE_DIM == 300

        # exclude neck and eyeball regions
        flame_joints_name = ('Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye')
        expr_vertex_idx = []
        flame_vertex_num = flame_2019['v_template'].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('Neck')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('L_Eye')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('R_Eye')] = 1
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)

        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]

        return expr_vertex_idx
    
    def get_arm(self, mesh_neutral_pose, skinning_weight):
        normal = Meshes(verts=mesh_neutral_pose[None,:,:], faces=torch.LongTensor(self.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(self.vertex_num_upsampled,3).detach()
        part_label = skinning_weight.argmax(1)
        is_arm = 0
        for name in ('R_Shoulder', 'R_Elbow', 'L_Shoulder', 'L_Elbow'):
            is_arm = is_arm + (part_label == self.joints_name.index(name))
        is_arm = (is_arm > 0)
        is_upper_arm = is_arm * (normal[:,1] > math.cos(math.pi/3))
        is_lower_arm = is_arm * (normal[:,1] <= math.cos(math.pi/3))
        return is_upper_arm, is_lower_arm


smpl_x = SMPLX()
