import os
import numpy as np
import torch
import pickle
import time
import importlib
import pdb
# from util.misc import save_img_np

CUDA = True

class Model(object):
    def __init__(self, mult_chan=None, depth=None, load_path=None, lr=0.0001,
                 nn_module='default_nn',
                 init_weights=True,
                 gpu_ids=0):
        
        self.criterion = torch.nn.MSELoss()  # TODO add overridable 
        
        if load_path is None:
            nn_name = nn_module
            nn_module = importlib.import_module('model_modules.nn_modules.' + nn_module)
            self.net = nn_module.Net()
            if CUDA:
                self.net.cuda()
            if init_weights:
                print("Initializing weights")
                self.net.apply(_weights_init)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999))
            self.meta = {
                'nn': nn_name,
                'count_iter': 0,
                'lr': lr
            }
        else:
            self.load_checkpoint(load_path)

        self.signal_v = None
        self.target_v = None

    def __str__(self):
        some_name = self.meta.get('nn')
        if some_name is None:   # TODO: remove once support for older models no longer needed
            some_name = self.meta.get('name')
        out_str = '{:s} | iter: {:d}'.format(some_name,
                                             self.meta['count_iter'])
        return out_str

    def save_checkpoint(self, save_path):
        """Save neural network and trainer states to disk."""
        time_start = time.time()
        training_state_dict = {
            'nn': self.net,
            'optimizer': self.optimizer,
            'meta_dict': self.meta
            }
        print('saving checkpoint to:', save_path)
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(training_state_dict, save_path)
        time_save = time.time() - time_start
        print('model save time: {:.1f} s'.format(time_save))

    def load_checkpoint(self, load_path):
        """Load neural network and trainer states from disk."""
        time_start = time.time()
        print('loading checkpoint from:', load_path)
        training_state_dict = torch.load(load_path)
        self.net = training_state_dict['nn']
        self.optimizer = training_state_dict['optimizer']
        self.meta = training_state_dict['meta_dict']
        time_load = time.time() - time_start
        print('model load time: {:.1f} s'.format(time_load))

    def set_lr(self, lr):
        lr_old = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = lr
        print('learning rate: {} => {}'.format(lr_old, lr))

    def do_train_iter(self, signal, target):
        self.net.train()

        if CUDA:
            self.signal_v = torch.autograd.Variable(torch.Tensor(signal).cuda())
            self.target_v = torch.autograd.Variable(torch.Tensor(target).cuda())
        else:
            self.signal_v = torch.autograd.Variable(torch.Tensor(signal))
            self.target_v = torch.autograd.Variable(torch.Tensor(target))
        # if self.signal_v is None:
        #     if CUDA:
        #         self.signal_v = torch.autograd.Variable(torch.Tensor(signal).cuda())
        #         self.target_v = torch.autograd.Variable(torch.Tensor(target).cuda())
        #     else:
        #         self.signal_v = torch.autograd.Variable(torch.Tensor(signal))
        #         self.target_v = torch.autograd.Variable(torch.Tensor(target))
        # else:
        #     self.signal_v.data.copy_(torch.Tensor(signal))
        #     self.target_v.data.copy_(torch.Tensor(target))
            
        self.optimizer.zero_grad()
        output = self.net(self.signal_v)
        loss = self.criterion(output, self.target_v)
        loss.backward()
        self.optimizer.step()
        # print("iter: {:3d} | loss: {:4f}".format(self.meta['count_iter'], loss.data[0]))
        self.meta['count_iter'] += 1
        return loss.data[0]
    
    def predict(self, signal):
        # print('{:s}: predicting {:d} examples'.format(self.meta['name'], signal.shape[0]))
        self.net.eval()
        if CUDA:
            signal_t = torch.Tensor(signal).cuda()
        else:
            signal_t = torch.Tensor(signal)
        signal_v = torch.autograd.Variable(signal_t, volatile=True)
        pred_v = self.net(signal_v)
        pred_np = pred_v.data.cpu().numpy()
        return pred_np

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 
