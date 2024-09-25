import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import eval_sh, RGB2SH, get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from smplx.lbs import batch_rigid_transform
#from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from pytorch3d.structures import Meshes
from config import cfg
import copy

class HumanGaussian(nn.Module):
    def __init__(self):
        super(HumanGaussian, self).__init__()
        self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        self.triplane_face = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())

        self.geo_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128], use_gn=True)
        self.mean_offset_net = make_linear_layers([128, 3], relu_final=False)
        self.scale_net = make_linear_layers([128, 1], relu_final=False)
        self.geo_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(smpl_x.joint_num-1)*6, 128, 128, 128], use_gn=True)
        self.mean_offset_offset_net = make_linear_layers([128, 3], relu_final=False)
        self.scale_offset_net = make_linear_layers([128, 1], relu_final=False)
        self.rgb_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128, 3], relu_final=False, use_gn=True)
        self.rgb_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(smpl_x.joint_num-1)*6+3, 128, 128, 128, 3], relu_final=False, use_gn=True)

        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender]).cuda()
        self.shape_param = nn.Parameter(smpl_x.shape_param.float().cuda())
        self.joint_offset = nn.Parameter(smpl_x.joint_offset.float().cuda())
     
    def init(self):
        # upsample mesh and other assets
        xyz, _, _, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        skinning_weight = self.smplx_layer.lbs_weights.float()
        pose_dirs = self.smplx_layer.posedirs.permute(1,0).reshape(smpl_x.vertex_num,3*(smpl_x.joint_num-1)*9)
        expr_dirs = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num,3*smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr = torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda()
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face[smpl_x.face_vertex_idx], is_face_expr[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0, 1.0
        is_cavity = torch.FloatTensor(smpl_x.is_cavity).cuda()[:,None]
        _, skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity]) # upsample with dummy vertex

        pose_dirs = pose_dirs.reshape(smpl_x.vertex_num_upsampled*3,(smpl_x.joint_num-1)*9).permute(1,0) 
        expr_dirs = expr_dirs.view(smpl_x.vertex_num_upsampled,3,smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr = is_rhand[:,0] > 0, is_lhand[:,0] > 0, is_face[:,0] > 0, is_face_expr[:,0] > 0
        is_cavity = is_cavity[:,0] > 0

        self.register_buffer('pos_enc_mesh', xyz)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('pose_dirs', pose_dirs)
        self.register_buffer('expr_dirs', expr_dirs)
        self.register_buffer('is_rhand', is_rhand)
        self.register_buffer('is_lhand', is_lhand)
        self.register_buffer('is_face', is_face)
        self.register_buffer('is_face_expr', is_face_expr)
        self.register_buffer('is_cavity', is_cavity)

    def get_optimizable_params(self):
        optimizable_params = [
            {'params': [self.triplane], 'name': 'triplane_human', 'lr': cfg.lr},
            {'params': [self.triplane_face], 'name': 'triplane_face_human', 'lr': cfg.lr},
            {'params': list(self.geo_net.parameters()), 'name': 'geo_net_human', 'lr': cfg.lr},
            {'params': list(self.mean_offset_net.parameters()), 'name': 'mean_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.scale_net.parameters()), 'name': 'scale_net_human', 'lr': cfg.lr},
            {'params': list(self.geo_offset_net.parameters()), 'name': 'geo_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.mean_offset_offset_net.parameters()), 'name': 'mean_offset_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.scale_offset_net.parameters()), 'name': 'scale_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.rgb_net.parameters()), 'name': 'rgb_net_human', 'lr': cfg.lr},
            {'params': list(self.rgb_offset_net.parameters()), 'name': 'rgb_offset_net_human', 'lr': cfg.lr},
            {'params': [self.shape_param], 'name': 'shape_param_human', 'lr': cfg.lr},
            {'params': [self.joint_offset], 'name': 'joint_offset_human', 'lr': cfg.lr}
        ]
        return optimizable_params

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(1,-1).cuda() # 大 pose
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        if use_id_info:
            shape_param = self.shape_param[None,:]
            face_offset = smpl_x.face_offset[None,:,:].float().cuda()
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
            #joint_offset = None
        else:
            shape_param = torch.zeros((1,smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        mesh_neutral_pose = output.vertices[0] # 大 pose human
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(mesh_neutral_pose) # 大 pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num,:] # 大 pose human

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(len(smpl_x.joint_part['body'])-1,3)
        zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']),3)
        neutral_body_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_body_pose)))
        jaw_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(jaw_pose)))
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None,:,:,:], joint_neutral_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_zero_pose_human(self, return_mesh=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint_part['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[None,:]
        face_offset = smpl_x.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        #joint_offset = None 
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        joint_zero_pose = output.joints[0][:smpl_x.joint_num,:] # zero pose human
        if not return_mesh:
            return joint_zero_pose
        else: 
            mesh_zero_pose = output.vertices[0] # zero pose human
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(mesh_zero_pose) # zero pose human
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        trans = smplx_param['trans'].view(1,3)

        # forward kinematics
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        
        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint
    
    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        skinning_weight = self.skinning_weight[nn_vertex_idxs,:]
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smpl_x.joint_num,16)).view(smpl_x.vertex_num_upsampled,4,4)
        return transform_mat_vertex

    def lbs(self, xyz, transform_mat_vertex, trans):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(smpl_x.vertex_num_upsampled,4)[:,:3]
        xyz = xyz + trans
        return xyz
    
    def extract_tri_feature(self):
        ## 1. triplane features of all vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        ## 2. triplane features of face vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh[self.is_face,:]
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_face_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_face_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_face_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane_face[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane_face[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane_face[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat_face = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # sum(self.is_face), cfg.triplane_shape[0]*3
        
        # combine 1 and 2
        tri_feat[self.is_face] = tri_feat_face
        return tri_feat

    def forward_geo_network(self, tri_feat, smplx_param):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)

        # combine pose with triplane feature
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose)).view(1,smpl_x.joint_num-1,6).repeat(smpl_x.vertex_num_upsampled,1,1) # without root pose
        pose = pose.view(smpl_x.vertex_num_upsampled, (smpl_x.joint_num-1)*6)
        feat = torch.cat((tri_feat, pose.detach()),1)

        # forward to geometry networks
        geo_offset_feat = self.geo_offset_net(feat)
        mean_offset_offset = self.mean_offset_offset_net(geo_offset_feat) # pose-dependent mean offset of Gaussians
        scale_offset = self.scale_offset_net(geo_offset_feat) # pose-dependent scale of Gaussians
        return mean_offset_offset, scale_offset
    
    def get_mean_offset_offset(self, smplx_param, mean_offset_offset):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) # without root pose

        # smplx pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(smpl_x.joint_num-1)*9)
        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(smpl_x.vertex_num_upsampled,3)

        # combine it with regressed mean_offset_offset
        # for face and hands, use smplx offset
        mask = ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)[:,None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        smplx_pose_offset = smplx_pose_offset * mask
        output = mean_offset_offset + smplx_pose_offset
        return output, mean_offset_offset

    def forward_rgb_network(self, tri_feat, smplx_param, xyz):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        
        # transform root pose from camera coordinate system to world coordinate system
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose))
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose))
        pose = pose.view(1,(smpl_x.joint_num-1)*6).repeat(smpl_x.vertex_num_upsampled,1) # smpl_x.vertex_num_upsampled, (smpl_x.joint_num-1)*6

        # per-vertex normal in world coordinate system
        with torch.no_grad():
            normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled,3)
            is_cavity = self.is_cavity[:,None].float()
            normal = normal * (1 - is_cavity) + (-normal) * is_cavity # cavity has opposite normal direction in the template mesh

        # forward to rgb network
        feat = torch.cat((tri_feat, pose.detach(), normal.detach()),1)
        rgb_offset = self.rgb_offset_net(feat) # pose-dependent rgb offset of Gaussians
        return rgb_offset

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

    def forward(self, smplx_param, cam_param=None, is_world_coord=True):
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = self.get_zero_pose_human()

        # extract triplane feature
        tri_feat = self.extract_tri_feature()
      
        # get Gaussian assets
        geo_feat = self.geo_net(tri_feat)
        mean_offset = self.mean_offset_net(geo_feat) # mean offset of Gaussians
        scale = self.scale_net(geo_feat) # scale of Gaussians
        rgb = self.rgb_net(tri_feat) # rgb of Gaussians
        mean_3d = mesh_neutral_pose + mean_offset # 大 pose
 
        # get pose-dependent Gaussian assets
        mean_offset_offset, scale_offset = self.forward_geo_network(tri_feat, smplx_param)
        scale, scale_refined = torch.exp(scale).repeat(1,3), torch.exp(scale+scale_offset).repeat(1,3)
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(smplx_param, mean_offset_offset)
        mean_3d_refined = mean_3d + mean_combined_offset # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.expr_dirs).sum(2)
        mean_3d = mean_3d + smplx_expr_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose

        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
        nn_vertex_idxs[mask] = torch.arange(smpl_x.vertex_num_upsampled).cuda()[mask]

        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs)
        mean_3d = self.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param

        # camera coordinate system -> world coordinate system
        if not is_world_coord:
            mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
            mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)

        # forward to rgb network
        rgb_offset = self.forward_rgb_network(tri_feat, smplx_param, mean_3d_refined)
        rgb, rgb_refined = (torch.tanh(rgb) + 1) / 2, (torch.tanh(rgb + rgb_offset) + 1) / 2 # normalize to [0,1]
        
        # Gaussians and offsets
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant rotation
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant opacity
        assets = {
                'mean_3d': mean_3d, 
                'opacity': opacity, 
                'scale': scale, 
                'rotation': rotation, 
                'rgb': rgb
                }
        assets_refined = {
                'mean_3d': mean_3d_refined, 
                'opacity': opacity, 
                'scale': scale_refined, 
                'rotation': rotation, 
                'rgb': rgb_refined
                }
        offsets = {
                'mean_offset': mean_offset,
                'mean_offset_offset': mean_offset_offset,
                'scale_offset': scale_offset,
                'rgb_offset': rgb_offset
                }
        return assets, assets_refined, offsets, mesh_neutral_pose

class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()
    
    def forward(self, gaussian_assets, img_shape, cam_param, bg=None):
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        scale = gaussian_assets['scale']
        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        if bg is None:
            bg = torch.ones((3)).float().cuda()
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg, 
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen). 
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None)
        
        return {'img': render_img,
                'depthmap': render_depthmap,
                'mask': render_mask,
                'mean_2d': mean_2d,
                'is_vis': radius > 0,
                'radius': radius}

class SMPLXParamDict(nn.Module):
    def __init__(self):
        super(SMPLXParamDict, self).__init__()

    # initialize SMPL-X parameters of all frames
    # used to train models from scratch
    def init(self, smplx_params):
        _smplx_params = {}
        for capture_id in smplx_params.keys():
            _smplx_params[capture_id] = nn.ParameterDict({})
            for frame_idx in smplx_params[capture_id].keys():
                _smplx_params[capture_id][str(frame_idx)] = nn.ParameterDict({})
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                    if 'pose' in param_name:
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(smplx_params[capture_id][frame_idx][param_name].cuda())))
                    else:
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(smplx_params[capture_id][frame_idx][param_name].cuda())
        self.smplx_params = nn.ParameterDict(_smplx_params)

    def get_optimizable_params(self):
        optimizable_params = []
        for capture_id in self.smplx_params.keys():
            for frame_idx in self.smplx_params[capture_id].keys():
                for param_name in self.smplx_params[capture_id][frame_idx].keys():
                    optimizable_params.append({'params': [self.smplx_params[capture_id][frame_idx][param_name]], 'name': 'smplx_' + param_name + '_' + capture_id + '_' + frame_idx, 'lr': cfg.smplx_param_lr})
        return optimizable_params

    def forward(self, capture_ids, frame_idxs):
        out = []
        for capture_id, frame_idx in zip(capture_ids, frame_idxs):
            capture_id = str(capture_id)
            frame_idx = str(int(frame_idx))
            smplx_param = {}
            for param_name in self.smplx_params[capture_id][frame_idx].keys():
                if 'pose' in param_name:
                    smplx_param[param_name] = matrix_to_axis_angle(rotation_6d_to_matrix(self.smplx_params[capture_id][frame_idx][param_name]))
                else:
                    smplx_param[param_name] = self.smplx_params[capture_id][frame_idx][param_name]
            out.append(smplx_param)
        return out
