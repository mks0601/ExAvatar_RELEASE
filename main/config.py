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
    warmup_itr = 100
    smplx_param_lr = 0 # for X-Humans dataset, we do not optimize smplx paraeters as it gives better results
    end_epoch = 25

    ## loss functions
    rgb_loss_weight = 0.8
    ssim_loss_weight = 0.2
    lpips_weight = 0.2

    ## dataset
    dataset = 'XHumans'

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

    def set_args(self, subject_id, continue_train=False):
        self.subject_id = subject_id
        self.continue_train = continue_train
        
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
