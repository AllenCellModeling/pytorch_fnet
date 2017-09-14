import os
import numpy as np
import torch
import pickle
import time
import importlib
import pdb
# from util.misc import save_img_np

class Model(object):
    def __init__(self, load_path=None, lr=0.0001,
                 nn_module='default_nn',
                 init_weights=True,
                 gpu_ids=0):
        
        self.criterion = torch.nn.L1Loss()
        if isinstance(gpu_ids, int):
            self._device_ids = [gpu_ids]
        else:
            assert isinstance(gpu_ids, (tuple, list))
            self._device_ids = gpu_ids
        
        if load_path is None:
            nn_name = nn_module
            nn_module = importlib.import_module('model_modules.nn_modules.' + nn_module)
            self.net = nn_module.Net()
            if self._device_ids[0] != -1:
                self.net = torch.nn.DataParallel(self.net, device_ids=self._device_ids)
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

            
    def __str__(self):
        some_name = self.meta.get('nn')
        if some_name is None:   # TODO: remove once support for older models no longer needed
            some_name = self.meta.get('name')
        out_str = '{:s} | L1 | iter: {:d}'.format(
            some_name,
            self.meta['count_iter'])
        return out_str

    def save_checkpoint(self, save_path):
        """Save neural network and trainer states to disk."""
        time_start = time.time()
        # self.net should be an instance of torch.nn.DataParallel
        module = self.net.module
        module.cpu()
        training_state_dict = {
            'nn': module,
            'optimizer': self.optimizer,
            'meta_dict': self.meta
            }
        print('saving checkpoint to:', save_path)
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(training_state_dict, save_path)
        module.cuda(self._device_ids[0])
        time_save = time.time() - time_start
        print('model save time: {:.1f} s'.format(time_save))

    def load_checkpoint(self, load_path):
        """Load neural network and trainer states from disk."""
        time_start = time.time()
        print('loading checkpoint from:', load_path)
        training_state_dict = torch.load(load_path)
        net_loaded = training_state_dict['nn']
        self.optimizer = training_state_dict['optimizer']
        self.meta = training_state_dict['meta_dict']
        if self._device_ids[0] == -1:
            self.net = net_loaded.cpu()
        else:
            self.net = torch.nn.DataParallel(net_loaded, device_ids=self._device_ids)
        self.optimizer.state = _set_gpu_recursive(self.optimizer.state, self._device_ids[0])
        time_load = time.time() - time_start
        print('model load time: {:.1f} s'.format(time_load))

    def set_lr(self, lr):
        lr_old = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = lr
        print('learning rate: {} => {}'.format(lr_old, lr))

    def do_train_iter(self, signal, target):
        self.net.train()
        if self._device_ids[0] != -1:
            signal_v = torch.autograd.Variable(torch.Tensor(signal).cuda(self._device_ids[0]))
            target_v = torch.autograd.Variable(torch.Tensor(target).cuda(self._device_ids[0]))
        else:
            signal_v = torch.autograd.Variable(torch.Tensor(signal))
            target_v = torch.autograd.Variable(torch.Tensor(target))
        self.optimizer.zero_grad()
        output = self.net(signal_v)
        loss = self.criterion(output, target_v)
        loss.backward()
        self.optimizer.step()
        # print("iter: {:3d} | loss: {:4f}".format(self.meta['count_iter'], loss.data[0]))
        self.meta['count_iter'] += 1
        return loss.data[0]
    
    def predict(self, signal):
        # time_start = time.time()
        self.net.eval()
        if self._device_ids[0] == -1:
            print('predicting on CPU')
            signal_t = torch.Tensor(signal)
        else:
            signal_t = torch.Tensor(signal).cuda()
        signal_v = torch.autograd.Variable(signal_t, volatile=True)
        pred_v = self.net(signal_v)
        pred_np = pred_v.data.cpu().numpy()
        # time_pred = time.time() - time_start
        # print('DEBUG: device {} predict time: {:.1f} s'.format(self._device_ids[0], time_pred))
        return pred_np

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 

# modified from pytorch_integrated_cell
def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.

    Parameters:
    var - (dict) keys are either Tensors or dicts
    gpu_id - (int) GPU onto which to move the Tensors
    """
    for key in var:
        if isinstance(var[key], dict):
            var[key] = _set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var  
