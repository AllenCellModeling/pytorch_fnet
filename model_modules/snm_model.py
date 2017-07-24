import os
import numpy as np
import torch
import pickle
import time
import importlib
import pdb

CUDA = True

class Model(object):
    def __init__(self, mult_chan=None, depth=None, load_path=None, lr=0.0001, nn_module=None, init_weights=True):
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        if load_path is None:
            nn_name = nn_module
            nn_module = importlib.import_module('model_modules.nn_modules.' + nn_module)
            self.net = nn_module.Net(mult_chan=mult_chan, depth=depth)
            if CUDA:
                self.net.cuda()
            if init_weights:
                print("Initializing weights")
                self.net.apply(_weights_init)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999))
            self.meta = {
                'nn': nn_name,
                'count_iter': 0,
                'mult_chan': mult_chan,
                'depth': depth,
                'lr': lr
            }
        else:
            self.load_checkpoint(load_path)

        self.signal_v = None
        self.target_v = None

    def __str__(self):
        some_name = self.meta.get('nn')
        out_str = '{:s} | lr: {:f} | iter: {:d}'.format(some_name,
                                                        self.meta['lr'],
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
        if self.signal_v is None:
            if CUDA:
                self.signal_v = torch.autograd.Variable(torch.Tensor(signal).cuda())
                self.target_v = torch.autograd.Variable(torch.Tensor(target).long().cuda())
            else:
                self.signal_v = torch.autograd.Variable(torch.Tensor(signal))
                self.target_v = torch.autograd.Variable(torch.Tensor(target).long())
        else:
            self.signal_v.data.copy_(torch.Tensor(signal))
            self.target_v.data.copy_(torch.Tensor(target).long())
            
        self.optimizer.zero_grad()
        output_pre = self.net(self.signal_v)
        # reshape nn output and target for CrossEntropy function
        output = output_pre.permute(0, 2, 3, 4, 1).contiguous().view(-1, 3)
        target_1d = self.target_v.view(-1)
        loss = self.criterion(output, target_1d)
        
        loss.backward()
        self.optimizer.step()
        self.meta['count_iter'] += 1
        return loss.data[0]
    
    def predict(self, signal):
        self.net.eval()
        if CUDA:
            signal_t = torch.Tensor(signal).cuda()
        else:
            signal_t = torch.Tensor(signal)
        signal_v = torch.autograd.Variable(signal_t)
        pred_v = self.net(signal_v)
        pred_np = pred_v.data.cpu().numpy()
        shape = (pred_np.shape[0], 1, *pred_np.shape[2:])
        pred = np.zeros(shape)
        pred[:, 0, :, :, :] = np.argmax(pred_np[:, :2, :, :, :], axis=1)
        # pred[:, 0, :, :, :] = np.argmax(pred_np[:, :3, :, :, :], axis=1)
        return pred

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 
