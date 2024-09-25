"""
prepare X-Humans dataset for training X-Avatar https://github.com/Skype-line/X-Avatar
1. to train the scan-based X-Avatar, generate 3D scans from .*pkl to *.ply/obj
2. to train the RGBD-based X-Avatar, generate colored PCL from RGB-D images
"""
import os
import os.path as osp
import glob
import pickle as pkl
import numpy as np
from tqdm import tqdm
import trimesh
from PIL import Image
import open3d as o3d
import cv2
import argparse
import sys

def convert_mesh_pkl2ply(data_root):
    """ Convert meshes from *.pkl to *.ply """
    sub_folder_list = sorted(os.listdir(data_root))
    for sub_folder in sub_folder_list:
        if not os.path.isdir(osp.join(data_root, sub_folder)):
            continue
        print('Start to process', sub_folder)
        mesh_file_list = sorted(
            glob.glob(
                os.path.join(data_root, sub_folder, 'meshes_pkl', 'mesh*.pkl')))
        save_folder = osp.join(data_root, sub_folder, 'meshes_ply')
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        for mesh_file in tqdm(mesh_file_list):
            uv_map_path = mesh_file.replace('mesh-', 'atlas-')
            mesh_data = pkl.load(open(mesh_file, 'rb'), encoding='latin1')
            mesh_data['uvs'] = np.clip(mesh_data['uvs'], a_min=0, a_max=1) # added

            if not os.path.exists(uv_map_path):
                print('uv map not exist: ', uv_map_path)
                continue
            uv_map = Image.fromarray(
                pkl.load(open(uv_map_path, 'rb'), encoding='latin1'))
            uv_map = uv_map.transpose(
                method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
            tex = trimesh.visual.texture.TextureVisuals(uv=mesh_data['uvs'],
                                                        image=uv_map)
            mesh_with_color = trimesh.Trimesh(
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces'],
                vertex_normals=mesh_data['normals'],
                visual=tex)
            mesh_with_color.visual = mesh_with_color.visual.to_color()
            mesh_with_color_name = osp.join(
                save_folder,
                mesh_file.split('/')[-1].replace('.pkl', '.ply'))
            _ = mesh_with_color.export(mesh_with_color_name)


def convert_mesh_pkl2obj(data_root):
    """Convert meshes from *.pkl to *.obj"""
    sub_folder_list = sorted(os.listdir(data_root))
    for sub_folder in sub_folder_list:
        if not os.path.isdir(osp.join(data_root, sub_folder)):
            continue
        print('Start to process', sub_folder)
        mesh_file_list = sorted(
            glob.glob(
                os.path.join(data_root, sub_folder, 'meshes_pkl', 'mesh*.pkl')))
        save_folder = osp.join(data_root, sub_folder, 'meshes_obj')
        os.makedirs(save_folder, exist_ok=True)
        for mesh_file in tqdm(mesh_file_list):
            mesh_data = pkl.load(open(mesh_file, 'rb'), encoding='latin1')
            mesh_name = osp.basename(mesh_file)
            obj_name = mesh_name.replace('.pkl', '.obj')
            mtl_name = mesh_name.replace('.pkl', '.mtl')
            texture_name = mesh_name.replace('.pkl', '.jpg').replace('mesh-', 'atlas-')
            # write obj
            with open(osp.join(save_folder, obj_name), 'w') as f:
                f.write("#OBJ\n")
                f.write(f"#{len(mesh_data['vertices'])} pos\n")
                for v in mesh_data['vertices']:
                    f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
                f.write(f"#{len(mesh_data['normals'])} norm\n")
                for vn in mesh_data['normals']:
                    f.write("vn %.4f %.4f %.4f\n" % (vn[0], vn[1], vn[2]))
                f.write(f"#{len(mesh_data['uvs'])} tex\n")
                for vt in mesh_data['uvs']:
                    f.write("vt %.4f %.4f\n" % (vt[0], vt[1]))
                f.write(f"#{len(mesh_data['faces'])} faces\n")
                f.write("mtllib {}\n".format(mtl_name))
                f.write("usemtl atlasTextureMap\n")
                for fc in mesh_data['faces']:
                    f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (fc[0]+1, fc[0]+1, fc[0]+1, fc[1]+1, fc[1]+1, fc[1]+1, fc[2]+1, fc[2]+1, fc[2]+1))
                
            # write mtl
            with open(osp.join(save_folder, mtl_name), 'w') as f:
                f.write("newmtl atlasTextureMap\n")
                s = 'map_Kd {}\n'.format(texture_name)  # map to image
                f.write(s)
            
            # write texture
            tex = pkl.load(open(mesh_file.replace('mesh-', 'atlas-'), 'rb'), encoding='latin1')
            uv_map = Image.fromarray(tex).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
            uv_map.save(osp.join(save_folder, texture_name))
            

def generate_pcl(data_root):
    print('Generating colored point clouds...')
    print('Loading {}'.format(data_root))
    sub_folder_list = sorted(os.listdir(data_root))
    for sub_folder in sub_folder_list:
        if not os.path.isdir(os.path.join(data_root, sub_folder)):
            continue
        print('Processing sequence: {}'.format(sub_folder))
        render_image_list = sorted(
            glob.glob(
                os.path.join(data_root, sub_folder, 'render', 'image',
                             '*.png')))
        render_depth_list = sorted(
            glob.glob(
                os.path.join(data_root, sub_folder, 'render', 'depth',
                         '*.tiff')))
        cam_infos = np.load(
            os.path.join(data_root, sub_folder, 'render', 'cameras.npz'))

        data_dir = os.path.join(data_root, sub_folder, 'pcl')
        os.makedirs(data_dir, exist_ok=True)
        for i in tqdm(range(len(render_image_list))):
            cam = o3d.camera.PinholeCameraIntrinsic()

            cam.intrinsic_matrix = cam_infos['intrinsic']
            cam_ext = cam_infos['extrinsic'][i]

            rgb_raw = cv2.imread(render_image_list[i])
            rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
            depth_raw = np.array(Image.open(render_depth_list[i]))
            depth_raw[depth_raw>10] = 0
            depth_raw = o3d.geometry.Image(depth_raw)
            full_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_raw,
                intrinsic=cam,
                extrinsic=cam_ext,
                depth_scale=1000,
                project_valid_depth_only=False)

            temp = np.array(full_pcd.points)
            temp = np.nan_to_num(temp)

            temp = temp.reshape((np.array(depth_raw).shape[0],
                                 np.array(depth_raw).shape[1], 3))

            valid_2d_pixels = np.array(
                np.where(np.any(temp, axis=2) == True)).transpose()

            temp = temp[valid_2d_pixels[:, 0], valid_2d_pixels[:, 1], :]
            temp = temp.reshape((-1, 3))
            tmp_color = rgb_raw[valid_2d_pixels[:, 0], valid_2d_pixels[:,
                                                                       1], :]
            tmp_color = tmp_color.reshape((-1, 3)) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(temp)
            pcd.colors = o3d.utility.Vector3dVector(tmp_color)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                  max_nn=30))

            cam_C = -np.linalg.inv(cam_ext[:3, :3]) @ cam_ext[:3, -1]
            center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            center_sphere.translate((cam_C[0], cam_C[1], cam_C[2]))

            o3d.geometry.PointCloud.orient_normals_towards_camera_location(
                pcd, camera_location=cam_C)
            # o3d.visualization.draw_geometries([pcd, center_sphere])
            frame_id = render_image_list[i].split('/')[-1].split('.')[0][-5:]
            o3d.io.write_point_cloud(
                os.path.join(data_dir, f"pcl-f{frame_id}.ply"), pcd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/X_Humans/00035')
    args = parser.parse_args()
    MODES = ['train', 'test']
    for mode in MODES:
        """ convert meshes to .ply format (by default) """
        convert_mesh_pkl2ply(osp.join(args.data_root, mode))
        """ convert meshes to .obj format"""
        # convert_mesh_pkl2obj(osp.join(args.data_root, mode))
        """ generate colored point clouds from RGB-D images """
        generate_pcl(osp.join(args.data_root, mode))

if __name__ == '__main__':
    sys.exit(main())
    

