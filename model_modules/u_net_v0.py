import os
import numpy as np
import torch
import torch.nn as nn
import pickle

GPU_ID = 0
CUDA = True

class Model(object):
    def __init__(self, mult_chan=None, depth=None, load_path=None):
        if load_path is None:
            self.net = Net(mult_chan=mult_chan, depth=depth)
            if CUDA:
                self.net.cuda()
            self.meta = {
                'name': 'U-Network V0',
                'count_iter': 0,
                'mult_chan': mult_chan,
                'depth': depth
            }
        else:
            self._load(load_path)  # defines self.net, self.meta

        lr = 0.0001
        momentum = 0.5
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.BCELoss()

    def __str__(self):
        out_str = '{:s} | mult_chan: {:d} | depth: {:d}'.format(self.meta['name'],
                                                                self.meta['mult_chan'],
                                                                self.meta['depth'])
        return out_str

    def save(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        package = (self.net, self.meta)
        with open(path, 'wb') as fo:
            print('saving model to:', path)
            pickle.dump(package, fo)
            print('saved model to:', path)

    def _load(self, path):
        print('loading model:', path)
        new_name = os.path.basename(path).split('.')[0]
        package = pickle.load(open(path, 'rb'))
        assert len(package) == 2
        self.net = package[0]
        self.meta = package[1]
        self.meta['name'] = new_name

    def do_train_iter(self, signal, target):
        self.net.train()
        if CUDA:
            signal_v = torch.autograd.Variable(torch.Tensor(signal).cuda())
            target_v = torch.autograd.Variable(torch.Tensor(target).cuda())
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
        print('{:s}: predicting {:d} examples'.format(self.meta['name'], signal.shape[0]))
        self.net.eval()
        if CUDA:
            signal_t = torch.Tensor(signal).cuda()
        else:
            signal_t = torch.Tensor(signal)
        signal_v = torch.autograd.Variable(signal_t)
        pred_v = self.net(signal_v)
        pred_np = pred_v.data.cpu().numpy()
        return pred_np

class Net(nn.Module):
    def __init__(self, mult_chan=16, depth=1):
        super().__init__()
        self.net_recurse = _Net_recurse(n_in_channels=1, mult_chan=mult_chan, depth=depth)
        self.conv_out = torch.nn.Conv3d(mult_chan,  1, kernel_size=3, padding=1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x_rec = self.net_recurse(x)
        x_pre_out = self.conv_out(x_rec)
        x_out = self.sig(x_pre_out)
        # return x_pre_out
        return x_out

class _Net_recurse(nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.p

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels)
            self.pool = torch.nn.MaxPool3d(2, stride=2)
            self.convt = torch.nn.ConvTranspose3d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_pool = self.pool(x_2conv_more)
            x_sub_u = self.sub_u(x_pool)
            x_convt = self.convt(x_sub_u)
            x_cat = torch.cat((x_2conv_more, x_convt), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv3d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
