import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.module import SceneGaussian, HumanGaussian, SMPLXParamDict, GaussianRenderer
from nets.layer import MeshRenderer
from nets.loss import RGBLoss, SSIM, LPIPS, LaplacianReg, JointOffsetSymmetricReg, HandMeanReg, HandRGBReg, ArmRGBReg
from utils.smpl_x import smpl_x
from utils.flame import flame
import copy
from config import cfg

class Model(nn.Module):
    def __init__(self, scene_gaussian, human_gaussian, smplx_param_dict):
        super(Model, self).__init__()
        self.scene_gaussian = scene_gaussian
        self.human_gaussian = human_gaussian
        self.smplx_param_dict = smplx_param_dict
        self.gaussian_renderer = GaussianRenderer()
        self.face_mesh_renderer = MeshRenderer(flame.vertex_uv, flame.face_uv)
        if cfg.fit_pose_to_test:
            self.optimizable_params = self.smplx_param_dict.get_optimizable_params()
        else:
            self.optimizable_params = self.scene_gaussian.get_optimizable_params() + self.human_gaussian.get_optimizable_params() 
            if smplx_param_dict is not None:
                self.optimizable_params += self.smplx_param_dict.get_optimizable_params()
        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender])
        self.rgb_loss = RGBLoss()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()
        self.hand_mean_reg = HandMeanReg()
        self.hand_rgb_reg = HandRGBReg()
        self.arm_rgb_reg = ArmRGBReg()
        self.eval_modules = [self.lpips]
    
    def get_smplx_outputs(self, smplx_param, cam_param):
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint_part['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint_part['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint_part['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)

        shape = self.human_gaussian.shape_param[None]
        face_offset = smpl_x.face_offset.cuda()[None]
        joint_offset = self.human_gaussian.joint_offset[None]

        # camera coordinate system
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        mesh = output.vertices[0]

        # camera coordinate system -> world coordinate system
        mesh = torch.matmul(torch.inverse(cam_param['R']), (mesh - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        return mesh

    def forward(self, data, cur_itr, mode):
        batch_size = data['cam_param']['R'].shape[0]
        img_height, img_width = data['img'].shape[2:]
        
        
        # in the training, increase SH degree following the schedule
        # in the testing stage, load and use it
        if mode == 'train':
            self.scene_gaussian.set_sh_degree(cur_itr)
        
        # background color for the human-only rendering
        if mode == 'train':
            bg = torch.rand(3).float().cuda()
        else:
            bg = torch.ones((3)).float().cuda()
           
        # get assets for the rendering and render
        human_assets, human_assets_refined, human_offsets, smplx_outputs = {}, {}, {}, []
        scene_renders, human_renders, scene_human_renders = {}, {}, {}
        human_renders_refined, scene_human_renders_refined = {}, {}
        face_renders, face_renders_refined = [], []
        for i in range(batch_size):
            
            # get assets form scene Gaussians
            scene_asset = self.scene_gaussian({k: v[i] for k,v in data['cam_param'].items()})
            
            # get assets and offsets from human Gaussians
            smplx_param = self.smplx_param_dict([data['frame_idx'][i]])[0]
            human_asset, human_asset_refined, human_offset, mesh_neutral_pose = self.human_gaussian(smplx_param, {k: v[i] for k,v in data['cam_param'].items()})
            
            # clamp scale in early of the training as garbage large scales from randomly initialized networks take HUGE GPU memory
            key_list = ['mean_3d', 'scale', 'rotation', 'rgb']
            if (mode == 'train') and cfg.is_warmup:
                human_asset['scale_wo_clamp'] = human_asset['scale'].clone()
                human_asset['scale'] = torch.clamp(human_asset['scale'], max=0.001)
                human_asset_refined['scale_wo_clamp'] = human_asset_refined['scale'].clone()
                human_asset_refined['scale'] = torch.clamp(human_asset_refined['scale'], max=0.001)
                key_list += ['scale_wo_clamp']

            # gather assets
            for key in key_list:
                if key not in human_assets:
                    human_assets[key] = [human_asset[key]]
                    human_assets_refined[key] = [human_asset_refined[key]]
                else:
                    human_assets[key].append(human_asset[key])
                    human_assets_refined[key].append(human_asset_refined[key])

            # gather offsets
            for key in ['mean_offset', 'mean_offset_offset', 'scale_offset', 'rgb_offset']:
                if key not in human_offsets:
                    human_offsets[key] = [human_offset[key]]
                else:
                    human_offsets[key].append(human_offset[key])
            
            # smplx outputs
            smplx_output = self.get_smplx_outputs(smplx_param, {k: v[i] for k,v in data['cam_param'].items()})
            smplx_outputs.append(smplx_output)
            
            # combine scene and human assets
            scene_human_asset = {}
            for key in ['mean_3d', 'opacity', 'scale', 'rotation', 'rgb']:
                scene_human_asset[key] = torch.cat((scene_asset[key].detach(), human_asset[key])) # detach scene

            # combine scene and human assets (refined)
            scene_human_asset_refined = {}
            for key in ['mean_3d', 'opacity', 'scale', 'rotation', 'rgb']:
                scene_human_asset_refined[key] = torch.cat((scene_asset[key].detach(), human_asset_refined[key])) # detach scene

            # scene render
            scene_render = self.gaussian_renderer(scene_asset, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()})
            for key in ['img', 'mean_2d', 'is_vis', 'radius']:
                if key not in scene_renders:
                    scene_renders[key] = [scene_render[key]]
                else:
                    scene_renders[key].append(scene_render[key])
            
            # human render
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            for key in ['img', 'mask']:
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])
            
            # scene and human render
            scene_human_render = self.gaussian_renderer(scene_human_asset, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()})
            for key in ['img']:
                if key not in scene_human_renders:
                    scene_human_renders[key] = [scene_human_render[key]]
                else:
                    scene_human_renders[key].append(scene_human_render[key])

            # human render (refined)
            human_render_refined = self.gaussian_renderer(human_asset_refined, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, bg)
            for key in ['img', 'mask']:
                if key not in human_renders_refined:
                    human_renders_refined[key] = [human_render_refined[key]]
                else:
                    human_renders_refined[key].append(human_render_refined[key])
            
            # scene and human render (refined)
            scene_human_render_refined = self.gaussian_renderer(scene_human_asset_refined, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()})
            for key in ['img']:
                if key not in scene_human_renders_refined:
                    scene_human_renders_refined[key] = [scene_human_render_refined[key]]
                else:
                    scene_human_renders_refined[key].append(scene_human_render_refined[key])
            
            # face render
            face_texture, face_texture_mask = flame.texture[None], flame.texture_mask[None,0:1]
            face_texture = torch.cat((face_texture, face_texture_mask),1)
            face_render = self.face_mesh_renderer(face_texture, human_asset['mean_3d'][None,smpl_x.face_vertex_idx,:], flame.face, {k: v[i,None] for k,v in data['cam_param'].items()}, (img_height, img_width)) 
            face_render_refined = self.face_mesh_renderer(face_texture, human_asset_refined['mean_3d'][None,smpl_x.face_vertex_idx,:], flame.face, {k: v[i,None] for k,v in data['cam_param'].items()}, (img_height, img_width)) 
            face_renders.append(face_render[0])
            face_renders_refined.append(face_render_refined[0])

        # aggregate assets and renders
        # do not perform any differentiable operations on mean_2d to get its gradients (we should make it the left node)
        human_assets = {k: torch.stack(v) for k,v in human_assets.items()}
        human_assets_refined = {k: torch.stack(v) for k,v in human_assets_refined.items()}
        human_offsets = {k: torch.stack(v) for k,v in human_offsets.items()}
        smplx_outputs = torch.stack(smplx_outputs)
        scene_renders = {k: (torch.stack(v) if k != 'mean_2d' else v) for k,v in scene_renders.items()}
        human_renders = {k: torch.stack(v) for k,v in human_renders.items()}
        scene_human_renders = {k: torch.stack(v) for k,v in scene_human_renders.items()}
        human_renders_refined = {k: torch.stack(v) for k,v in human_renders_refined.items()}
        scene_human_renders_refined = {k: torch.stack(v) for k,v in scene_human_renders_refined.items()}
        face_renders = torch.stack(face_renders)
        face_renders_refined = torch.stack(face_renders_refined)
      
        if mode == 'train':
            # track stats to densify and prune Gaussians
            stats = {'mean_2d': scene_renders['mean_2d'], 'is_vis': scene_renders['is_vis'].detach(), 'radius': scene_renders['radius'].detach()}

            # loss functions
            loss = {}
            loss['rgb_human'] = self.rgb_loss(scene_human_renders['img'], data['img'], bbox=data['bbox']) * cfg.rgb_loss_weight
            loss['ssim_human'] = (1 - self.ssim(scene_human_renders['img'], data['img'], bbox=data['bbox'])) * cfg.ssim_loss_weight
            loss['lpips_human'] = self.lpips(scene_human_renders['img'], data['img'], bbox=data['bbox']) * cfg.lpips_weight
            is_face = ((face_renders[:,:3] != -1) * (face_renders[:,3:] == 1)).float()
            loss['rgb_face'] = self.rgb_loss(scene_human_renders['img'] * (1 - is_face) + face_renders[:,:3] * is_face, data['img'], bbox=data['bbox']) * cfg.rgb_loss_weight
            loss['rgb_human_rand_bg'] = self.rgb_loss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])
            
            loss['rgb_human_refined'] = self.rgb_loss(scene_human_renders_refined['img'], data['img'], bbox=data['bbox']) * cfg.rgb_loss_weight
            loss['ssim_human_refined'] = (1 - self.ssim(scene_human_renders_refined['img'], data['img'], bbox=data['bbox'])) * cfg.ssim_loss_weight
            loss['lpips_human_refined'] = self.lpips(scene_human_renders_refined['img'], data['img'], bbox=data['bbox']) * cfg.lpips_weight
            is_face = ((face_renders_refined[:,:3] != -1) * (face_renders_refined[:,3:] == 1)).float()
            loss['rgb_face_refined'] = self.rgb_loss(scene_human_renders_refined['img'] * (1 - is_face) + face_renders_refined[:,:3] * is_face, data['img'], bbox=data['bbox']) * cfg.rgb_loss_weight
            loss['rgb_human_refined_rand_bg'] = self.rgb_loss(human_renders_refined['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])

            if cfg.fit_pose_to_test:
                return stats, loss

            loss['rgb_scene'] = self.rgb_loss(scene_renders['img'], data['img']) * (1 - data['mask']) * cfg.rgb_loss_weight
            loss['ssim_scene'] = (1 - self.ssim(scene_renders['img'], data['img'], mask=1-data['mask'])) * cfg.ssim_loss_weight

            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda() * 10
            weight[:,self.human_gaussian.is_rhand,:] = 1000
            weight[:,self.human_gaussian.is_lhand,:] = 1000
            weight[:,self.human_gaussian.is_face,:] = 1
            weight[:,self.human_gaussian.is_face_expr,:] = 10
            loss['gaussian_mean_reg'] = (human_offsets['mean_offset'] ** 2 + human_offsets['mean_offset_offset'] ** 2) * weight
            loss['gaussian_mean_hand_reg'] = self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand) + self.hand_mean_reg(mesh_neutral_pose, human_offsets['mean_offset_offset'], self.human_gaussian.is_lhand, self.human_gaussian.is_rhand)
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_rhand,:] = 1000 
            weight[:,self.human_gaussian.is_lhand,:] = 1000
            weight[:,self.human_gaussian.is_face_expr,:] = 10
            weight[:,self.human_gaussian.is_cavity,:] = 0
            if cfg.is_warmup:
                loss['gaussian_scale_reg'] = (human_assets['scale_wo_clamp'] ** 2 + human_offsets['scale_offset'] ** 2) * weight
            else:
                loss['gaussian_scale_reg'] = (human_assets['scale'] ** 2 + human_offsets['scale_offset'] ** 2) * weight
            
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda()
            weight[:,self.human_gaussian.is_face_expr,:] = 50
            weight[:,self.human_gaussian.is_cavity,:] = 0.1
            loss['lap_mean'] = (self.lap_reg(mesh_neutral_pose[None,:,:].detach() + human_offsets['mean_offset'], mesh_neutral_pose[None,:,:].detach()) + \
                                self.lap_reg(mesh_neutral_pose[None,:,:].detach() + human_offsets['mean_offset'] + human_offsets['mean_offset_offset'], mesh_neutral_pose[None,:,:].detach())) * 100000 * weight
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda() * 10 # changed from 1
            weight[:,self.human_gaussian.is_rhand,:] = 10
            weight[:,self.human_gaussian.is_lhand,:] = 10
            weight[:,self.human_gaussian.is_face_expr,:] = 0
            loss['lap_scale'] = (self.lap_reg(human_assets['scale'], None) + self.lap_reg(human_assets_refined['scale'], None)) * 100000 * weight
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda() * 0.1
            weight[:,self.human_gaussian.is_rhand,:] = 100
            weight[:,self.human_gaussian.is_lhand,:] = 100
            loss['lap_rgb'] = (self.lap_reg(human_assets['rgb'], None) + self.lap_reg(human_assets_refined['rgb'], None)) * weight

            loss['hand_rgb_reg'] = (self.hand_rgb_reg(human_assets['rgb'], self.human_gaussian.is_rhand, self.human_gaussian.is_lhand) + self.hand_rgb_reg(human_assets_refined['rgb'], self.human_gaussian.is_rhand, self.human_gaussian.is_lhand)) * 0.01
            is_upper_arm, is_lower_arm = smpl_x.get_arm(mesh_neutral_pose, self.human_gaussian.skinning_weight)
            loss['arm_rgb_reg'] = (self.arm_rgb_reg(mesh_neutral_pose, is_upper_arm, is_lower_arm, human_assets['rgb']) + self.arm_rgb_reg(mesh_neutral_pose, is_upper_arm, is_lower_arm, human_assets_refined['rgb'])) * 0.1

            weight = torch.ones((smpl_x.joint_num,3)).float().cuda()
            weight[smpl_x.joint_part['lhand'],:] = 10
            weight[smpl_x.joint_part['rhand'],:] = 10
            loss['joint_offset_reg'] = (self.human_gaussian.joint_offset - smpl_x.joint_offset.cuda()) ** 2 * weight
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(self.human_gaussian.joint_offset)
            return stats, loss
        else:
            out = {}
            out['scene_img'] = scene_renders['img']
            out['human_img'] = human_renders['img']
            out['scene_human_img'] = scene_human_renders['img']
            out['human_img_refined'] = human_renders_refined['img']
            out['scene_human_img_refined'] = scene_human_renders_refined['img']
            out['smplx_mesh'] = smplx_outputs

            is_face = (face_renders[:,:3] != -1).float() * face_renders[:,3:]
            out['human_face_img'] = human_renders['img'] * (1 - is_face) + face_renders[:,:3] * is_face
            is_face = (face_renders_refined[:,:3] != -1).float() * face_renders_refined[:,3:]
            out['human_face_img_refined'] = human_renders_refined['img'] * (1 - is_face) + face_renders_refined[:,:3] * is_face
            
            is_fg = human_renders['mask'] > 0.9
            out['scene_human_img_composed'] = is_fg * human_renders['img'] + (1 - is_fg.float()) * scene_human_renders['img']
            is_fg = human_renders_refined['mask'] > 0.9
            out['scene_human_img_refined_composed'] = is_fg * human_renders_refined['img'] + (1 - is_fg.float()) * scene_human_renders_refined['img']
            return out
    
    def adjust_gaussians(self, track, gaussian, cur_itr, optimizer):
        batch_size = track['mean_2d_grad'].shape[0]
        
        # keep track of max radius in image-space for pruning
        for i in range(batch_size):
            gaussian.radius_max[track['is_vis'][i]] = torch.maximum(gaussian.radius_max[track['is_vis'][i]], track['radius'][i][track['is_vis'][i]])
            gaussian.track_stats(track['mean_2d_grad'][i], track['is_vis'][i])

        if (cur_itr > cfg.densify_start_itr) and (cur_itr % cfg.densify_interval == 0):
            size_threshold = 20 if cur_itr > cfg.opacity_reset_interval else None
            gaussian.densify_and_prune(size_threshold, optimizer)
        
        if (cur_itr > 0) and (cur_itr % cfg.opacity_reset_interval == 0):
            gaussian.reset_opacity(optimizer)
    
def get_model(scene, cam_dist, scene_point_num, smplx_params):
    scene_gaussian = SceneGaussian()
    with torch.no_grad():
        if (scene is not None) and (scene_point_num is None):
            scene_gaussian.init_from_point_cloud(scene[:,:3], scene[:,3:], cam_dist)
        elif (scene is None) and (scene_point_num is not None):
            scene_gaussian.init_from_point_num(scene_point_num)
        else:
            assert 0, 'Unsupported initialization'

    human_gaussian = HumanGaussian()
    with torch.no_grad():
        human_gaussian.init()
    
    if smplx_params is not None:
        smplx_param_dict = SMPLXParamDict()
        with torch.no_grad():
            smplx_param_dict.init(smplx_params)
    else:
        smplx_param_dict = None

    model = Model(scene_gaussian, human_gaussian, smplx_param_dict)
    return model
