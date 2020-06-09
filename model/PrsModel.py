import torch
from . import network
import os
from util.util import print_network

class ClassPrsModel:
    """ Class for training Model weights
    :args opt: structure containing configuration params
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        #

        # load/define networks
        self.net = network.define_prs(self.gpu_ids)
        self.net.train(self.is_train)
        self.criterion1 = network.define_loss_1(opt)
        self.criterion2 = network.define_loss_2(opt)


        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = network.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_sample = data
        # set inputs
        self.sample_ = input_sample
        self.voxel = data['voxel']
        self.points = data['points']
        self.nstvoxel = data['closeset']

    def forward(self):
        out = self.net(self.voxel)
        # self.voxel 4*1*32*32*32
        return out

    def backward(self, out):
        # loss_sd_ = torch.sum(loss_sd) / opt.batch_size
        self.loss1 = self.criterion1(out, self.sample_)
        self.loss1_ = torch.sum(self.loss1) / self.opt.batch_size
        # print('loss1', self.loss1_)
        self.loss2 = self.criterion2(out)
        self.loss2_ = torch.sum(self.loss2) / self.opt.batch_size
        # print('loss2', self.loss2_)
        self.loss = self.loss1_ + self.opt.Regular_loss*self.loss2_
        print('loss', self.loss)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = os.path.join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pkl' % (which_epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

