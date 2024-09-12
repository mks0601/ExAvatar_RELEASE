import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layer import XY2UV, get_face_index_map_uv
from nets.loss import CoordLoss, PoseLoss, LaplacianReg, EdgeLengthLoss, FaceOffsetSymmetricReg, JointOffsetSymmetricReg
from utils.smpl_x import smpl_x
from utils.flame import flame
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import math
from config import cfg

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.xy2uv = XY2UV(flame.vertex_uv, flame.face_uv, cfg.uvmap_shape)
        self.smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
        self.flame_layer = copy.deepcopy(flame.layer).cuda()
        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()
        self.lap_reg = LaplacianReg(flame.vertex_num, flame.face)
        self.edge_length_loss = EdgeLengthLoss(flame.face)
        self.face_offset_sym_reg = FaceOffsetSymmetricReg()
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()

    def process_input_smplx_param(self, smplx_param):
        out = {}

        # rotation 6d -> axis angle
        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(smplx_param[key]))

        # others
        out['trans'] = smplx_param['trans']
        out['expr'] = smplx_param['expr']
        out['shape'] = smplx_param['shape']
        out['face_offset'] = smplx_param['face_offset']
        out['joint_offset'] = smplx_param['joint_offset']
        out['locator_offset'] = smplx_param['locator_offset']
        return out

    def process_input_flame_param(self, flame_param):
        out = {}

        # rotation 6d -> axis angle
        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(flame_param[key]))

        # others
        out['trans'] = flame_param['trans']
        out['expr'] = flame_param['expr']
        out['shape'] = flame_param['shape']
        return out

    def get_smplx_coord(self, smplx_param, cam_param, use_pose=True, use_expr=True, use_face_offset=True, use_joint_offset=True, use_locator_offset=True, root_rel=False):
        batch_size = smplx_param['root_pose'].shape[0]
       
        if use_pose:
            root_pose = smplx_param['root_pose']
            body_pose = smplx_param['body_pose']
            jaw_pose = smplx_param['jaw_pose']
            leye_pose = smplx_param['leye_pose']
            reye_pose = smplx_param['reye_pose']
            lhand_pose = smplx_param['lhand_pose']
            rhand_pose = smplx_param['rhand_pose']
        else:
            root_pose = torch.zeros_like(smplx_param['root_pose'])
            body_pose = torch.zeros_like(smplx_param['body_pose'])
            jaw_pose = torch.zeros_like(smplx_param['jaw_pose'])
            leye_pose = torch.zeros_like(smplx_param['leye_pose'])
            reye_pose = torch.zeros_like(smplx_param['reye_pose'])
            lhand_pose = torch.zeros_like(smplx_param['lhand_pose'])
            rhand_pose = torch.zeros_like(smplx_param['rhand_pose'])
        
        if use_expr:
            expr = smplx_param['expr']
        else:
            expr = torch.zeros_like(smplx_param['expr'])

        if use_face_offset:
            face_offset = smpl_x.get_face_offset(smplx_param['face_offset'])
        else:
            face_offset = None
 
        if use_joint_offset:
            joint_offset = smpl_x.get_joint_offset(smplx_param['joint_offset'])
        else:
            joint_offset = None
       
        if use_locator_offset:
            locator_offset = smpl_x.get_locator_offset(smplx_param['locator_offset'])
        else:
            locator_offset = None

        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose.detach(), leye_pose=leye_pose.detach(), reye_pose=reye_pose.detach(), left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr.detach(), betas=smplx_param['shape'], face_offset=face_offset, joint_offset=joint_offset, locator_offset=locator_offset) # detach jaw_pose, leye_pose, reye_pose, and expr as they are optimized by flame layer

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
        root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + smplx_param['trans'][:,None,:]
        kpt_cam = kpt_cam - root_cam[:,None,:] + smplx_param['trans'][:,None,:]
        
        # project to the 2D space
        if cam_param is not None:
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2)

        if root_rel:
            mesh_cam = mesh_cam - smplx_param['trans'][:,None,:]
            kpt_cam = kpt_cam - smplx_param['trans'][:,None,:]

        if cam_param is not None:
            return mesh_cam, kpt_cam, root_cam, kpt_proj
        else:
             return mesh_cam, kpt_cam, root_cam

    def get_flame_coord(self, flame_param, cam_param, use_pose=True, use_expr=True):
        if use_pose:
            root_pose = flame_param['root_pose']
            neck_pose = flame_param['neck_pose']
            jaw_pose = flame_param['jaw_pose']
            leye_pose = flame_param['leye_pose']
            reye_pose = flame_param['reye_pose']
        else:
            root_pose = torch.zeros_like(flame_param['root_pose'])
            neck_pose = torch.zeros_like(flame_param['neck_pose'])
            jaw_pose = torch.zeros_like(flame_param['jaw_pose'])
            leye_pose = torch.zeros_like(flame_param['leye_pose'])
            reye_pose = torch.zeros_like(flame_param['reye_pose'])
        
        if use_expr:
            expr = flame_param['expr']
        else:
            expr = torch.zeros_like(flame_param['expr'])

        output = self.flame_layer(global_orient=root_pose, neck_pose=neck_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, expression=expr, betas=flame_param['shape'])

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        kpt_cam = output.joints
        lear = mesh_cam[:,flame.lear_vertex_idx,:]
        rear = mesh_cam[:,flame.rear_vertex_idx,:]
        kpt_cam = torch.cat((kpt_cam, lear[:,None,:], rear[:,None,:]),1) # follow flame.kpt['name']
        root_cam = kpt_cam[:,flame.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + flame_param['trans'][:,None,:]
        kpt_cam = kpt_cam - root_cam[:,None,:] + flame_param['trans'][:,None,:]
        
        if cam_param is not None:
            # project to the 2D space
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2) 
            return mesh_cam, kpt_cam, kpt_proj
        else:
            return mesh_cam, kpt_cam
    
    def check_face_visibility(self, face_mesh, leye, reye):
        center = face_mesh.mean(1)
        eye = (leye + reye)/2.

        eye_vec = eye - center
        cam_vec = center - 0

        eye_vec = F.normalize(torch.stack((eye_vec[:,0], eye_vec[:,2]),1), p=2, dim=1)
        cam_vec = F.normalize(torch.stack((cam_vec[:,0], cam_vec[:,2]),1), p=2, dim=1)

        dot_prod = torch.sum(eye_vec * cam_vec, 1)
        face_valid = dot_prod < math.cos(math.pi/4*3)
        return face_valid

    def get_smplx_full_pose(self, smplx_param):
        pose = torch.cat((smplx_param['root_pose'][:,None,:], smplx_param['body_pose'], smplx_param['jaw_pose'][:,None,:], smplx_param['leye_pose'][:,None,:], smplx_param['reye_pose'][:,None,:], smplx_param['lhand_pose'], smplx_param['rhand_pose']),1) # follow smpl_x.joint['name']
        return pose
 
    def get_flame_full_pose(self, flame_param):
        pose = torch.cat((flame_param['neck_pose'][:,None,:], flame_param['jaw_pose'][:,None,:], flame_param['leye_pose'][:,None,:], flame_param['reye_pose'][:,None,:]),1) # follow flame.joint['name'] without the root joint
        return pose
   
    def forward(self, smplx_inputs, flame_inputs, data, return_output):
        smplx_inputs = self.process_input_smplx_param(smplx_inputs) 
        flame_inputs = self.process_input_flame_param(flame_inputs) 
       
        # get coordinates from optimizable parameters
        smplx_mesh_cam, smplx_kpt_cam, smplx_root_cam, smplx_kpt_proj = self.get_smplx_coord(smplx_inputs, data['cam_param_proj'])
        smplx_mesh_cam_wo_fo, smplx_kpt_cam_wo_fo, smplx_root_cam_wo_fo, smplx_kpt_proj_wo_fo = self.get_smplx_coord(smplx_inputs, data['cam_param_proj'], use_face_offset=False)
        flame_mesh_cam, flame_kpt_cam, flame_kpt_proj = self.get_flame_coord(flame_inputs, data['cam_param_proj'])
        
        # get coordinates from zero pose
        smplx_mesh_wo_pose_wo_expr, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_pose=False, use_expr=False, use_locator_offset=False, root_rel=True)
        flame_mesh_wo_pose_wo_expr, _ = self.get_flame_coord(flame_inputs, cam_param=None, use_pose=False, use_expr=False)

        with torch.no_grad():
            # get coordinates from the initial parameters
            data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
            data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
            data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
            data['smplx_param']['joint_offset'] = smplx_inputs['joint_offset'].clone().detach()
            data['smplx_param']['locator_offset'] = smplx_inputs['locator_offset'].clone().detach()
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = self.get_smplx_coord(data['smplx_param'], data['cam_param_proj'], use_face_offset=False)

            # check face visibility
            face_valid = self.check_face_visibility(smplx_mesh_cam_init[:,smpl_x.face_vertex_idx,:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('L_Eye'),:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('R_Eye'),:])
            face_valid = face_valid * data['flame_valid']

        # loss functions
        loss = {}
        weight = torch.ones_like(smplx_kpt_proj)
        if not cfg.warmup:
            weight[:,[i for i in range(smpl_x.kpt['num']) if 'Face' in smpl_x.kpt['name'][i]],:] = 0
            weight[face_valid,:,:] = 1 # do not use 2D loss if face is not visible
        loss['smplx_kpt_proj'] = self.coord_loss(smplx_kpt_proj, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach()) * weight
        loss['smplx_kpt_proj_wo_fo'] = self.coord_loss(smplx_kpt_proj_wo_fo, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach()) * weight
        loss['flame_kpt_proj'] = torch.abs(flame_kpt_proj - data['kpt_img'][:,smpl_x.kpt['part_idx']['face'],:]) * data['kpt_valid'][:,smpl_x.kpt['part_idx']['face'],:] * weight[:,smpl_x.kpt['part_idx']['face'],:]
        if cfg.warmup:
            loss['flame_to_smplx_v2v'] = torch.abs(flame_mesh_cam - smplx_mesh_cam[:,smpl_x.face_vertex_idx,:].detach())
        else:
            loss['smplx_shape_reg'] = smplx_inputs['shape'] ** 2 * 0.01
            loss['smplx_mesh'] = torch.abs((smplx_mesh_cam_wo_fo - smplx_kpt_cam_wo_fo[:,smpl_x.kpt['root_idx'],None,:]) - \
                                            (smplx_mesh_cam_init - smplx_kpt_cam_init[:,smpl_x.kpt['root_idx'],None,:])) * 0.1 
            smplx_input_pose = self.get_smplx_full_pose(smplx_inputs)
            smplx_init_pose = self.get_smplx_full_pose(data['smplx_param'])
            loss['smplx_pose'] = self.pose_loss(smplx_input_pose, smplx_init_pose) * 0.1
            loss['smplx_pose_reg'] = torch.stack([smplx_input_pose[:,i,0] for i in range(smpl_x.joint['num']) if smpl_x.joint['name'][i] in ['Spine_1', 'Spine_2', 'Spine_3', 'Neck', 'Head']],1) ** 2  # prevent forward head posture

            flame_input_pose = self.get_flame_full_pose(flame_inputs)
            flame_init_pose = self.get_flame_full_pose(data['flame_param'])
            loss['flame_pose'] = self.pose_loss(flame_input_pose, flame_init_pose) * 0.1
            loss['flame_shape'] = torch.abs(flame_inputs['shape'] - data['flame_param']['shape']) * 0.1
            loss['flame_expr'] = torch.abs(flame_inputs['expr'] - data['flame_param']['expr']) * 0.1

            is_not_neck = torch.ones((1,flame.vertex_num,1)).float().cuda()
            is_not_neck[:,flame.layer.lbs_weights.argmax(1)==flame.joint['root_idx'],:] = 0
            loss['smplx_to_flame_v2v_wo_pose_expr'] = torch.abs(\
                    (smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:] - smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:].mean(1)[:,None,:]) - \
                    (flame_mesh_wo_pose_wo_expr - flame_mesh_wo_pose_wo_expr.mean(1)[:,None,:]).detach()) * is_not_neck * 10
            loss['smplx_to_flame_lap'] = self.lap_reg(smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:], flame_mesh_wo_pose_wo_expr.detach()) * is_not_neck * 100000
            loss['smplx_to_flame_edge_length'] = self.edge_length_loss(smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:], flame_mesh_wo_pose_wo_expr.detach(), is_not_neck)

            is_neck = torch.zeros((1,flame.vertex_num,1)).float().cuda()
            is_neck[:,flame.layer.lbs_weights.argmax(1)==flame.joint['root_idx'],:] = 1
            loss['face_offset_reg'] = smplx_inputs['face_offset'] ** 2 * is_neck * 1000
            weight = torch.ones((1,smpl_x.joint['num'],1)).float().cuda()
            if not cfg.hand_joint_offset:
                weight[:,smpl_x.joint['part_idx']['lhand'],:] = 10
                weight[:,smpl_x.joint['part_idx']['rhand'],:] = 10
            loss['joint_offset_reg'] = smplx_inputs['joint_offset'] ** 2 * 100 * weight
            loss['locator_offset_reg'] = smplx_inputs['locator_offset'] ** 2
            loss['face_offset_sym_reg'] = self.face_offset_sym_reg(smplx_inputs['face_offset'])
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(smplx_inputs['joint_offset'])
            loss['locator_offset_sym_reg'] = self.joint_offset_sym_reg(smplx_inputs['locator_offset'])
        
        if not return_output:
            return loss, None
        else:
            # for the visualization
            smplx_mesh_cam_wo_jo, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_joint_offset=False)
            smplx_mesh_wo_pose_wo_expr_wo_fo, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_pose=False, use_expr=False, use_face_offset=False, use_locator_offset=False, root_rel=True)

            # translation alignment
            offset = -flame_mesh_wo_pose_wo_expr.mean(1) + smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:].mean(1)
            flame_mesh_wo_pose_wo_expr = flame_mesh_wo_pose_wo_expr + offset[:,None,:]
            
            # face unwrap to uv space
            face_texture, face_texture_mask = self.xy2uv(data['img_face'], flame_mesh_cam, flame.face, data['cam_param_face'])
            face_texture = face_texture * flame.uv_mask[None,None,:,:] * face_valid[:,None,None,None]
            face_texture_mask = face_texture_mask * flame.uv_mask[None,None,:,:] * face_valid[:,None,None,None]

            # outputs
            out = {}
            out['smplx_mesh_cam'] = smplx_mesh_cam
            out['smplx_mesh_cam_wo_jo'] = smplx_mesh_cam_wo_jo
            out['smplx_mesh_cam_wo_fo'] = smplx_mesh_cam_wo_fo
            out['smplx_trans'] = smplx_inputs['trans'] - smplx_root_cam
            out['flame_mesh_cam'] = flame_mesh_cam
            out['smplx_mesh_wo_pose_wo_expr'] = smplx_mesh_wo_pose_wo_expr
            out['smplx_mesh_wo_pose_wo_expr_wo_fo'] = smplx_mesh_wo_pose_wo_expr_wo_fo
            out['flame_mesh_wo_pose_wo_expr'] = flame_mesh_wo_pose_wo_expr
            out['face_texture'] = face_texture
            out['face_texture_mask'] = face_texture_mask
            return loss, out
 
def get_model():
    model = Model()
    return model
