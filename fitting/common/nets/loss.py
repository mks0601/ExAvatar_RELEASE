import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils.smpl_x import smpl_x
from pytorch3d.transforms import axis_angle_to_matrix
from config import cfg

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()
 
    def get_bbox(self, kpt_proj, kpt_valid, extend_ratio=1.2):
        x, y = kpt_proj[kpt_valid[:,0]>0,0], kpt_proj[kpt_valid[:,0]>0,1]
        xmin, ymin = torch.min(x), torch.min(y)
        xmax, ymax = torch.max(x), torch.max(y)

        x_center = (xmin+xmax)/2.; width = xmax-xmin;
        xmin = x_center - 0.5 * width * extend_ratio
        xmax = x_center + 0.5 * width * extend_ratio
        
        y_center = (ymin+ymax)/2.; height = ymax-ymin;
        ymin = y_center - 0.5 * height * extend_ratio
        ymax = y_center + 0.5 * height * extend_ratio
        
        bbox = torch.FloatTensor([xmin, ymin, xmax-xmin, ymax-ymin]).cuda()
        return bbox
    
    def get_iou(self, box1, box2):
        box1 = box1.clone()
        box2 = box2.clone()
        box1[2:] += box1[:2] # xywh -> xyxy
        box2[2:] += box2[:2] # xywh -> xyxy

        xmin = torch.maximum(box1[0], box2[0])
        ymin = torch.maximum(box1[1], box2[1])
        xmax = torch.minimum(box1[2], box2[2])
        ymax = torch.minimum(box1[3], box2[3])
        inter_area = torch.maximum(torch.zeros_like(xmax-xmin), xmax-xmin) * torch.maximum(torch.zeros_like(ymax-ymin), ymax-ymin)
     
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / (union_area + 1e-5)
        return iou
   
    def forward(self, kpt_proj, kpt_proj_gt, kpt_valid, kpt_cam):
        weight = torch.ones_like(kpt_proj)

        # if boxes of two hands have high iou, ignore hands with bigger depth
        # 2D keypoint detector often gets confused between left and right hands when one hand is occluded by the other hand
        with torch.no_grad():
            batch_size = weight.shape[0]
            for i in range(batch_size):
                if (kpt_valid[i,smpl_x.kpt['part_idx']['lhand'],:].sum() == 0) or (kpt_valid[i,smpl_x.kpt['part_idx']['rhand'],:].sum()) == 0:
                    continue
                lhand_bbox = self.get_bbox(kpt_proj[i,smpl_x.kpt['part_idx']['lhand'],:], kpt_valid[i,smpl_x.kpt['part_idx']['lhand'],:])
                rhand_bbox = self.get_bbox(kpt_proj[i,smpl_x.kpt['part_idx']['rhand'],:], kpt_valid[i,smpl_x.kpt['part_idx']['rhand'],:])
                iou = self.get_iou(lhand_bbox, rhand_bbox)

                if float(iou) > 0.5:
                    if kpt_cam[i,smpl_x.kpt['part_idx']['lhand'],2].mean() > kpt_cam[i,smpl_x.kpt['part_idx']['rhand'],2].mean():
                        weight[i,smpl_x.kpt['part_idx']['lhand'],:] = 0
                        weight[i,smpl_x.kpt['name'].index('L_Wrist'),:] = 0
                    else:
                        weight[i,smpl_x.kpt['part_idx']['rhand'],:] = 0
                        weight[i,smpl_x.kpt['name'].index('R_Wrist'),:] = 0
        
        loss = torch.abs(kpt_proj - kpt_proj_gt) * kpt_valid * weight
        return loss

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt):
        batch_size = pose_out.shape[0]

        pose_out = pose_out.view(batch_size,-1,3)
        pose_gt = pose_gt.view(batch_size,-1,3)
        
        pose_out = axis_angle_to_matrix(pose_out)
        pose_gt = axis_angle_to_matrix(pose_gt)

        loss = torch.abs(pose_out - pose_gt)
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

class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True))

        valid_mask_1 = valid[:,face[:,0],:] * valid[:,face[:,1],:]
        valid_mask_2 = valid[:,face[:,0],:] * valid[:,face[:,2],:]
        valid_mask_3 = valid[:,face[:,1],:] * valid[:,face[:,2],:]

        diff1 = torch.abs(d1_out - d1_gt) * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt) * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt) * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3),1)
        return loss

class FaceOffsetSymmetricReg(nn.Module):
    def __init__(self):
        super(FaceOffsetSymmetricReg, self).__init__()
    
    def forward(self, face_offset):
        batch_size = face_offset.shape[0]
        _face_offset = torch.zeros((batch_size,smpl_x.vertex_num,3)).float().cuda()
        _face_offset[:,smpl_x.face_vertex_idx,:] = face_offset
        face_offset = _face_offset

        closest_faces = torch.LongTensor(smpl_x.flip_corr['closest_faces'].astype(np.int64)).cuda()
        bc = torch.FloatTensor(smpl_x.flip_corr['bc']).cuda()
        face_offset_flip = torch.sum(face_offset[:,closest_faces.view(-1),:].view(batch_size,smpl_x.vertex_num,3,3) * bc.view(1,smpl_x.vertex_num,3,1), 2)

        loss = torch.abs(face_offset[:,:,0] + face_offset_flip[:,:,0]) + torch.abs(face_offset[:,:,1] - face_offset_flip[:,:,1]) + torch.abs(face_offset[:,:,2] - face_offset_flip[:,:,2])
        loss = loss[:,smpl_x.face_vertex_idx]
        return loss

class JointOffsetSymmetricReg(nn.Module):
    def __init__(self):
        super(JointOffsetSymmetricReg, self).__init__()
    
    def forward(self, joint_offset):
        right_joint_idx, left_joint_idx = [], []
        for j in range(smpl_x.joint['num']):
            if smpl_x.joint['name'][j][:2] == 'R_':
                right_joint_idx.append(j)
                idx = smpl_x.joint['name'].index('L_' + smpl_x.joint['name'][j][2:])
                left_joint_idx.append(idx)

        loss = torch.abs(joint_offset[:,right_joint_idx,0] + joint_offset[:,left_joint_idx,0]) + torch.abs(joint_offset[:,right_joint_idx,1] - joint_offset[:,left_joint_idx,1]) + torch.abs(joint_offset[:,right_joint_idx,2] - joint_offset[:,left_joint_idx,2])
        return loss

