import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from config import cfg
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_model

# dynamic dataset import
exec('from ' + cfg.dataset + ' import ' + cfg.dataset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, optimizable_parameters):
        self.optimizer = torch.optim.Adam(optimizable_parameters, lr=cfg.lr)

    def set_lr(self, itr):
        if len(cfg.lr_dec_itr) == 0:
            return

        for e in cfg.lr_dec_itr:
            if itr < e:
                break
        if itr < cfg.lr_dec_itr[-1]:
            idx = cfg.lr_dec_itr.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_itr))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        self.trainset_loader = eval(cfg.dataset)(transforms.ToTensor())
        self.itr_per_epoch = math.ceil(len(self.trainset_loader) / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = DataLoader(dataset=self.trainset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=True, num_workers=cfg.num_thread)
        self.smplx_params = self.trainset_loader.smplx_params
        self.flame_params = self.trainset_loader.flame_params
        self.flame_shape_param = self.trainset_loader.flame_shape_param

    def _make_model(self, epoch=None):
        model = get_model()
        model = DataParallel(model).cuda()
        model.eval()
        self.model = model
