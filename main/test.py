import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import cv2
from utils.smpl_x import smpl_x
from pytorch3d.io import save_obj

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
 
    for itr, data in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(data, None, 'test')
        
        # save
        human_img = out['human_img'].cpu().numpy()
        human_img_refined = out['human_img_refined'].cpu().numpy()
        human_face_img = out['human_face_img'].cpu().numpy()
        human_face_img_refined = out['human_face_img_refined'].cpu().numpy()
        smplx_mesh = out['smplx_mesh'].cpu().numpy()
        batch_size = human_img.shape[0]
        for i in range(batch_size):
            capture_id = str(data['capture_id'][i])
            frame_idx = int(data['frame_idx'][i])
            save_root_path = osp.join(cfg.result_dir, 'test', capture_id)
            os.makedirs(save_root_path, exist_ok=True)

            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human.png'), human_img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_refined.png'), human_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_face.png'), human_face_img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, str(frame_idx) + '_human_face_refined.png'), human_face_img_refined[i].transpose(1,2,0)[:,:,::-1]*255)

            save_obj(osp.join(save_root_path, str(frame_idx) + '_smplx.obj'), torch.FloatTensor(smplx_mesh[i]), torch.LongTensor(smpl_x.face))

    
if __name__ == "__main__":
    main()
