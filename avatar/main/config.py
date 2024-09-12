import os
import os.path as osp
import sys

class Config:
    
    ## shape
    triplane_shape_3d = (2, 2, 2)
    triplane_face_shape_3d = (0.3, 0.3, 0.3)
    triplane_shape = (32, 128, 128)
  
    ## train
    lr = 1e-3 
    end_epoch = 5
    max_sh_degree = 3
    increase_sh_degree_interval = 1000
    densify_end_itr = 15000
    densify_start_itr = 500
    densify_interval = 100
    opacity_reset_interval = 3000
    densify_grad_thr = 0.0002
    opacity_min = 0.005
    dense_percent_thr = 0.01
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scale_lr = 0.005
    rotation_lr = 0.001
    warmup_itr = 100

    ## loss functions
    rgb_loss_weight = 0.8
    ssim_loss_weight = 0.2
    lpips_weight = 0.2

    ## dataset
    dataset = 'NeuMan' # Custom, NeuMan

    ## others
    num_thread = 8
    num_gpus = 1
    batch_size = 1 # Gaussian splatting renderer only supports batch_size==1
    smplx_gender = 'male' # only use male version as female version is not very good

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

    def set_args(self, subject_id, fit_pose_to_test=False, continue_train=False):
        self.subject_id = subject_id
        self.fit_pose_to_test = fit_pose_to_test
        self.continue_train = continue_train
        if self.fit_pose_to_test:
            self.smplx_param_lr = 1e-3
            self.model_dir = osp.join(self.model_dir, subject_id + '_fit_pose_to_test')
            self.result_dir = osp.join(self.result_dir, subject_id + '_fit_pose_to_test')
        else:
            self.smplx_param_lr = 1e-4
            self.model_dir = osp.join(self.model_dir, subject_id)
            self.result_dir = osp.join(self.result_dir, subject_id)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def set_stage(self, itr):
        if itr < self.warmup_itr:
            self.is_warmup = True
        else:
            self.is_warmup = False
    
cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
