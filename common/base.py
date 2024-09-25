import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
import numpy as np
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

    def get_optimizer(self, optimizable_params):
        optimizer = torch.optim.Adam(optimizable_params, lr=0.0, eps=1e-15)
        return optimizer
    
    def set_lr(self, cur_itr, tot_itr):
        for param_group in self.optimizer.param_groups:
            if 'human' in param_group['name']:
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = cfg.lr / 10
                elif (cur_itr > 0.95 * tot_itr):
                    param_group['lr'] = cfg.lr / 100
            elif 'smplx' in param_group['name']:
                lr = cfg.smplx_param_lr
                if 'hand' in param_group['name']:
                    lr = cfg.smplx_hand_param_lr
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = lr / 10
                elif (cur_itr > 0.95 * tot_itr):
                    param_group['lr'] = lr / 100

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_loader = eval(cfg.dataset)(transforms.ToTensor(), 'train')
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)
        self.smplx_params = trainset_loader.smplx_params

    def _make_model(self, epoch=None):
        if cfg.continue_train:
            ckpt = self.load_model()
            model = get_model(self.smplx_params)
            model = DataParallel(model).cuda()
            model.module.load_state_dict(ckpt['network'], strict=False)
        else:
            model = get_model(self.smplx_params)
            model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model.module.optimizable_params)

        if cfg.continue_train:
            start_epoch = ckpt['epoch'] + 1
            optimizer.load_state_dict(ckpt['optimizer'], strict=False)
        else:
            start_epoch = 0
        model.train()
        for module in model.module.eval_modules:
            module.eval()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth')]) for file_name in model_file_list])
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location='cpu')
        return ckpt

class Tester(Base):
    def __init__(self, test_epoch):
        super(Tester, self).__init__(log_name = 'test_logs.txt')
        self.test_epoch = int(test_epoch)

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.dataset)(transforms.ToTensor(), 'test')
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator
        self.smplx_params = testset_loader.smplx_params

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
       
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.smplx_params)
        model = DataParallel(model).cuda()
        model.module.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

