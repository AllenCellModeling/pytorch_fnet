import numpy as np
import torch
import torch.nn as nn
import pickle

GPU_ID = 0
CUDA = True

class Model(object):
    def __init__(self, mult_chan, depth):
        self.name = 'U-Network V0'
        self.mult_chan = mult_chan
        self.depth = depth
        self.net = Net(mult_chan=mult_chan, depth=depth)
        # self.net = Net_bk(mult_chan)
        # print(self.net)
        if CUDA:
            self.net.cuda()

        lr = 0.0001
        momentum = 0.5
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.BCELoss()
        self.count_iter = 0

    def __str__(self):
        out_str = '{:s} | mult_chan: {:d} | depth: {:d}'.format(self.name, self.mult_chan, self.depth)
        return out_str

    def save(self, fname):
        raise NotImplementedError
        print('saving model to:', fname)
        package = (self.net, self.mean_features, self.std_features)
        fo = open(fname, 'wb')
        pickle.dump(package, fo)
        fo.close()

    def load(self, fname):
        raise NotImplementedError
        print('loading model:', fname)
        classifier_tup = pickle.load(open(fname, 'rb'))
        self.net = classifier_tup[0]
        self.mean_features = classifier_tup[1]
        self.std_features = classifier_tup[2]

    def do_train_iter(self, signal, target):
        self.net.train()
        if CUDA:
            signal_t, target_t = torch.Tensor(signal).cuda(), torch.Tensor(target).cuda()
        else:
            signal_t, target_t = torch.Tensor(signal), torch.Tensor(target)
        signal_v, target_v = torch.autograd.Variable(signal_t), torch.autograd.Variable(target_t)
        self.optimizer.zero_grad()
        output = self.net(signal_v)
        loss = self.criterion(output, target_v)
        
        loss.backward()
        self.optimizer.step()
        print("iter: {:3d} | loss: {:4f}".format(self.count_iter, loss.data[0]))
        self.count_iter += 1
    
    def score(self, x):
        print('{:s}: scoring {:d} examples'.format(self.name, x.shape[0]))
        features_pp = (x - self.mean_features)/self.std_features
        self.net.eval()
        features_pp_v = torch.autograd.Variable(torch.FloatTensor(features_pp)).cuda(GPU_ID)
        scores_v = self.net(features_pp_v)
        scores_np = scores_v.data.cpu().numpy()
        return scores_np

    def predict(self, signal):
        print('{:s}: predicting {:d} examples'.format(self.name, signal.shape[0]))
        self.net.eval()
        if CUDA:
            signal_t = torch.Tensor(signal).cuda()
        else:
            signal_t = torch.Tensor(signal)
        signal_v = torch.autograd.Variable(signal_t)
        pred_v = self.net(signal_v)
        pred_np = pred_v.data.cpu().numpy()
        return pred_np

class Net_bk(nn.Module):
    def __init__(self, param_1=16):
        super().__init__()
        self.sub_1 = SubNet2Conv(1, param_1)
        self.pool_1 = torch.nn.MaxPool3d(2, stride=2)
        self.sub_2 = SubNet2Conv(param_1, param_1*2)
        self.convt = torch.nn.ConvTranspose3d(param_1*2, param_1, kernel_size=2, stride=2)
        self.sub_3 = SubNet2Conv(param_1*2, param_1)
        self.conv_out = torch.nn.Conv3d(param_1,  1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = self.sub_1(x)
        x1d = self.pool_1(x1)
        x2 = self.sub_2(x1d)
        x2u = self.convt(x2)  # upsample
        x1_2 = torch.cat((x1, x2u), 1)  # concatenate
        x3 = self.sub_3(x1_2)
        x_out = self.conv_out(x3)
        return x_out

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
