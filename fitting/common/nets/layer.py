import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras, RasterizationSettings, MeshRasterizer
import numpy as np
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

def get_face_index_map_xy(mesh, face, cam_param, render_shape):
    batch_size = mesh.shape[0]
    face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    outputs = rasterizer(mesh)
    return outputs

class XY2UV(nn.Module):
    def __init__(self, vertex_uv, face_uv, uvmap_shape):
        super(XY2UV, self).__init__()
        vertex_uv = torch.from_numpy(vertex_uv).float().cuda()[None,:,:]
        face_uv = torch.from_numpy(face_uv).long().cuda()[None,:,:]
        outputs = get_face_index_map_uv(vertex_uv, face_uv, uvmap_shape)
        pix_to_face = outputs.pix_to_face # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel. invalid: -1
        bary_coords = outputs.bary_coords # batch_size, uvmap_shape[0], uvmap_shape[1], faces_per_pixel, 3. invalid: -1
        self.pix_to_face_uv = pix_to_face[0,:,:,0]
        self.bary_coords_uv = bary_coords[0,:,:,0,:]
        self.uvmap_shape = uvmap_shape

    def forward(self, img, mesh_cam, face, cam_param):
        batch_size, channel_dim, img_height, img_width = img.shape

        # project mesh
        x = mesh_cam[:,:,0] / mesh_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
        y = mesh_cam[:,:,1] / mesh_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
        mesh_img = torch.stack((x,y),2)

        # get visible faces from mesh
        outputs = get_face_index_map_xy(mesh_cam, face, cam_param, (img_height, img_width))
        pix_to_face = outputs.pix_to_face # batch_size, img_height, img_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation!
        
        # get 2D coordinates of visible vertices
        mesh_img_0, mesh_img_1, mesh_img_2, invisible_uv = [], [], [], []
        for i in range(batch_size):
            # get visible face idxs
            visible_faces = torch.unique(pix_to_face_xy[i])
            visible_faces[visible_faces != -1] = visible_faces[visible_faces != -1] - face.shape[0] * i # packed -> unpacked
            valid_face_mask = torch.zeros((face.shape[0])).float().cuda()
            valid_face_mask[visible_faces] = 1.0
            
            # mask idxs of invisible vertices to -1
            _face = torch.from_numpy(face).cuda()
            _face[valid_face_mask==0,:] = -1
            vertex_idx_0 = _face[self.pix_to_face_uv.view(-1),0].view(self.uvmap_shape[0],self.uvmap_shape[1])
            vertex_idx_1 = _face[self.pix_to_face_uv.view(-1),1].view(self.uvmap_shape[0],self.uvmap_shape[1])
            vertex_idx_2 = _face[self.pix_to_face_uv.view(-1),2].view(self.uvmap_shape[0],self.uvmap_shape[1])
            invisible_uv.append((vertex_idx_0 == -1))
            
            # get 2D coordinates
            mesh_img_0.append(mesh_img[i,vertex_idx_0.view(-1),:].view(self.uvmap_shape[0], self.uvmap_shape[1], 2))
            mesh_img_1.append(mesh_img[i,vertex_idx_1.view(-1),:].view(self.uvmap_shape[0], self.uvmap_shape[1], 2))
            mesh_img_2.append(mesh_img[i,vertex_idx_2.view(-1),:].view(self.uvmap_shape[0], self.uvmap_shape[1], 2))
        mesh_img_0 = torch.stack(mesh_img_0) # batch_size, self.uvmap_shape[0], self.uvmap_shape[1], 2
        mesh_img_1 = torch.stack(mesh_img_1)
        mesh_img_2 = torch.stack(mesh_img_2)
        invisible_uv = torch.stack(invisible_uv).float() # batch_size, self.uvmap_shape[0], self.uvmap_shape[1]

        # prepare coordinates to perform grid_sample
        mesh_img = mesh_img_0 * self.bary_coords_uv[None,:,:,0,None] + mesh_img_1 * self.bary_coords_uv[None,:,:,1,None] + mesh_img_2 * self.bary_coords_uv[None,:,:,2,None]
        mesh_img = torch.stack((mesh_img[:,:,:,0]/(img_width-1)*2-1, mesh_img[:,:,:,1]/(img_height-1)*2-1), 3) # [-1,1] normalization
       
        # fill UV map
        uvmap = F.grid_sample(img, mesh_img, align_corners=True)
        uvmap[self.pix_to_face_uv[None,None,:,:].repeat(batch_size,channel_dim,1,1) == -1] = -1
        uvmap = uvmap * (1 - invisible_uv)[:,None,:,:] - invisible_uv[:,None,:,:]
        mask = uvmap != -1 # fg: 1, bg: 0
        uvmap = uvmap * mask
        return uvmap, mask

