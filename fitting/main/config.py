import os
import os.path as osp
import sys

class Config:
    
    ## fitting
    face_img_shape = (256, 256)
    proj_shape = (8, 8)
    uvmap_shape = (512, 512)
    lr_dec_factor = 10
    end_epoch = 3
    batch_size = 64
    body_3d_size = 2 # meter

    ## dataset
    dataset = 'Custom' # 'NeuMan', 'Custom', 'XHumans'

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join('..', 'common', 'utils', 'human_model_files')

    def set_args(self, subject_id):
        self.subject_id = subject_id
    
    def set_itr_opt_num(self, epoch):
        if epoch == 0:
            self.itr_opt_num = 500
        else:
            self.itr_opt_num = 250

    def set_stage(self, epoch, itr):
        if epoch == 0:
            self.lr = 1e-1
            self.lr_dec_itr = [100, 250, 400]
            self.stage_itr = [100, 250]
            if itr < self.stage_itr[0]:
                self.warmup = True
                self.hand_joint_offset = False
            elif itr < self.stage_itr[1]:
                self.warmup = False
            else:
                self.hand_joint_offset = True
        else:
            self.lr = 1e-2
            self.lr_dec_itr = [100, 200]
       
cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
