import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer import TexturesUV
from config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_gn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_gn:
                layers.append(nn.GroupNorm(4, feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


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

class MeshRenderer(nn.Module):
    def __init__(self, vertex_uv, face_uv):
        super(MeshRenderer, self).__init__()
        self.vertex_uv = torch.FloatTensor(vertex_uv).cuda()
        self.face_uv = torch.LongTensor(face_uv).cuda()

    def forward(self, uvmap, mesh, face, cam_param, render_shape):
        batch_size, uvmap_dim, uvmap_height, uvmap_width = uvmap.shape
        render_height, render_width = render_shape

        # get visible faces from mesh
        mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        fragments = get_face_index_map_xy(mesh, face, cam_param, (render_height, render_width))
        vertex_uv = torch.stack((self.vertex_uv[:,0], 1 - self.vertex_uv[:,1]),1)[None,:,:].repeat(batch_size,1,1) # flip y-axis following PyTorch3D convention
        renderer = TexturesUV(uvmap.permute(0,2,3,1), self.face_uv[None,:,:].repeat(batch_size,1,1), vertex_uv)
        render = renderer.sample_textures(fragments) # batch_size, render_height, render_width, faces_per_pixel, uvmap_dim
        render = render[:,:,:,0,:].permute(0,3,1,2) # batch_size, uvmap_dim, render_height, render_width
        
        # fg mask
        pix_to_face = fragments.pix_to_face # batch_size, render_height, render_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation

        # packed -> unpacked
        is_valid = (pix_to_face_xy != -1).float()
        pix_to_face_xy = (pix_to_face_xy - torch.arange(batch_size)[:,None,None].cuda() * face.shape[0]) * is_valid + (-1) * (1 - is_valid)
        pix_to_face_xy = pix_to_face_xy.long()
        
        # make backgroud pixels to -1
        render[pix_to_face_xy[:,None,:,:].repeat(1,uvmap_dim,1,1) == -1] = -1
        return render


