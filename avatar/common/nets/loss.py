import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import lpips
from utils.smpl_x import smpl_x
from pytorch3d.structures import Meshes
from config import cfg

class RGBLoss(nn.Module):
    def __init__(self):
        super(RGBLoss, self).__init__()
    
    def forward(self, img_out, img_target, bbox=None, mask=None, bg=None):
        if (mask is not None) and (bg is not None):
            img_target = img_target * mask + (1 - mask) * bg[:,:,None,None]
        if bbox is not None:
            img_height, img_width = img_out.shape[2:]
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        loss = torch.abs(img_out - img_target)
        return loss

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.FloatTensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
        return gauss / gauss.sum()

    def create_window(self, window_size, feat_dim):
        window_1d = self.gaussian(window_size, 1.5)[:,None]
        window_2d = torch.mm(window_1d, window_1d.permute(1,0))[None,None,:,:]
        window_2d = window_2d.repeat(feat_dim,1,1,1)
        return window_2d

    def forward(self, img_out, img_target, bbox=None, mask=None, window_size=11):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if mask is not None:
            img_out = img_out * mask
            img_target = img_target * mask
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]

        window = self.create_window(window_size, feat_dim)
        mu1 = F.conv2d(img_out, window, padding=window_size//2, groups=feat_dim)
        mu2 = F.conv2d(img_target, window, padding=window_size//2, groups=feat_dim)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img_out*img_out, window, padding=window_size//2, groups=feat_dim) - mu1_sq
        sigma2_sq = F.conv2d(img_target*img_target, window, padding=window_size//2, groups=feat_dim) - mu2_sq
        sigma1_sigma2 = F.conv2d(img_out*img_target, window, padding=window_size//2, groups=feat_dim) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sigma2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

# image perceptual loss (LPIPS. https://github.com/richzhang/PerceptualSimilarity)
class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').cuda()

    def forward(self, img_out, img_target, bbox=None):
        batch_size, feat_dim, img_height, img_width = img_out.shape
        if bbox is not None:
            xmin, ymin, width, height = [int(x) for x in bbox[0]]
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmin+width, img_width)
            ymax = min(ymin+height, img_height)
            img_out = img_out[:,:,ymin:ymax,xmin:xmax]
            img_target = img_target[:,:,ymin:ymax,xmin:xmax]
        img_out = img_out * 2 - 1 # [0,1] -> [-1,1]
        img_target = img_target * 2 - 1 # [0,1] -> [-1,1]
        loss = self.lpips(img_out, img_target)
        return loss

class LaplacianReg(nn.Module):
    def __init__(self, vertex_num, face):
        super(LaplacianReg, self).__init__()
        self.neighbor_idxs, self.neighbor_weights = self.get_neighbor(vertex_num, face)

    def get_neighbor(self, vertex_num, face, neighbor_max_num = 10):
        adj = {i: set() for i in range(vertex_num)}
        for i in range(len(face)):
            for idx in face[i]:
                adj[idx] |= set(face[i]) - set([idx])

        neighbor_idxs = np.tile(np.arange(vertex_num)[:,None], (1, neighbor_max_num))
        neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
        for idx in range(vertex_num):
            neighbor_num = min(len(adj[idx]), neighbor_max_num)
            neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
            neighbor_weights[idx,:neighbor_num] = -1.0 / neighbor_num
        
        neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs).cuda(), torch.from_numpy(neighbor_weights).cuda()
        return neighbor_idxs, neighbor_weights
    
    def compute_laplacian(self, x, neighbor_idxs, neighbor_weights):
        lap = x + (x[:, neighbor_idxs] * neighbor_weights[None, :, :, None]).sum(2)
        return lap

    def forward(self, out, target):
        if target is None:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            loss = lap_out ** 2
            return loss
        else:
            lap_out = self.compute_laplacian(out, self.neighbor_idxs, self.neighbor_weights)
            lap_target = self.compute_laplacian(target, self.neighbor_idxs, self.neighbor_weights)
            loss = (lap_out - lap_target) ** 2
            return loss

class JointOffsetSymmetricReg(nn.Module):
    def __init__(self):
        super(JointOffsetSymmetricReg, self).__init__()
    
    def forward(self, joint_offset):
        right_joint_idx, left_joint_idx = [], []
        for j in range(smpl_x.joint_num):
            if smpl_x.joints_name[j][:2] == 'R_':
                right_joint_idx.append(j)
                idx = smpl_x.joints_name.index('L_' + smpl_x.joints_name[j][2:])
                left_joint_idx.append(idx)

        loss = torch.abs(joint_offset[right_joint_idx,0] + joint_offset[left_joint_idx,0]) + torch.abs(joint_offset[right_joint_idx,1] - joint_offset[left_joint_idx,1]) + torch.abs(joint_offset[right_joint_idx,2] - joint_offset[left_joint_idx,2])
        return loss

class HandMeanReg(nn.Module):
    def __init__(self):
        super(HandMeanReg, self).__init__()
 
    def forward(self, mesh_neutral_pose, offset, is_rhand, is_lhand):
        batch_size = offset.shape[0]
        is_hand = (is_rhand + is_lhand) > 0
        with torch.no_grad():
            normal = Meshes(verts=mesh_neutral_pose[None,:,:], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(1,smpl_x.vertex_num_upsampled,3).detach().repeat(batch_size,1,1)
        dot_prod = torch.sum(normal * F.normalize(offset, p=2, dim=2), 2)[:,is_hand]
        loss = torch.clamp(dot_prod, min=0)
        return loss

class HandRGBReg(nn.Module):
    def __init__(self):
        super(HandRGBReg, self).__init__()
 
    def forward(self, rgb, is_rhand, is_lhand):
        rhand_rgb_out = rgb[:,is_rhand,:]
        lhand_rgb_out = rgb[:,is_lhand,:]
        rhand_rgb_target = rgb[:,is_rhand,:].mean(1)[:,None,:].detach()
        lhand_rgb_target = rgb[:,is_lhand,:].mean(1)[:,None,:].detach()
        loss = (rhand_rgb_out - rhand_rgb_target) ** 2 + (lhand_rgb_out - lhand_rgb_target) ** 2
        return loss

class ArmRGBReg(nn.Module):
    def __init__(self):
        super(ArmRGBReg, self).__init__()
 
    def forward(self, mesh_neutral_pose, is_upper_arm, is_lower_arm, rgb):
        batch_size = rgb.shape[0]
        
        # measure x-axis distance
        mesh_upper_arm = mesh_neutral_pose[is_upper_arm,:]
        mesh_lower_arm = mesh_neutral_pose[is_lower_arm,:]
        dist_x = torch.abs(mesh_lower_arm[:,None,0] - mesh_upper_arm[None,:,0])
        
        #
        dist_x_thr = 0.01 
        dist_x_mask = (dist_x < dist_x_thr).float()
        valid_num = int(torch.min(dist_x_mask.sum(1)))
        dist = torch.sqrt(torch.sum((mesh_lower_arm[:,None,:] - mesh_upper_arm[None,:,:]) ** 2, 2))
        dist = dist * dist_x_mask + 9999 * (1 - dist_x_mask)

        # pick top_k vertices with shortest x-axis distance from upper arm and average color
        top_k = min(50, valid_num)
        idxs = torch.topk(dist, k=top_k, dim=1, largest=False, sorted=False)[1]
        upper_arm_idx = torch.arange(smpl_x.vertex_num_upsampled).long().cuda()[is_upper_arm][idxs].view(-1)
        loss = (rgb[:,is_lower_arm,:] - rgb[:,upper_arm_idx,:].view(batch_size,int(is_lower_arm.sum()),top_k,3).mean(2).detach()) ** 2
        return loss
       
