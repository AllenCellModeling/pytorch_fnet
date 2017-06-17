import numpy as np
import torch
import torch.nn as nn
# import torch.utils.data
# import pickle

GPU_ID = 0
CUDA = True

class Model(object):
    def __init__(self):
        self.name = 'chek model'
        print('ctor:', self.name)
        self.net = Net()
        if CUDA:
            self.net.cuda()

        lr = 0.01
        momentum = 0.5
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.criterion = torch.nn.MSELoss()

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

    def _split_data(self, x, y, portion_test=0.1):
        idx = int((1 - portion_test)*x.shape[0])
        x_train = x[0:idx]
        x_val = x[idx:]
        y_train = y[0:idx]
        y_val = y[idx:]
        return x_train, y_train, x_val, y_val

    def do_train_iter(self, signal, target):
        self.net.train()
        if CUDA:
            signal_t, target_t = torch.Tensor(signal).cuda(), torch.Tensor(target).cuda()
        else:
            signal_t, target_t = torch.Tensor(signal), torch.Tensor(target)
        signal_v, target_v = torch.autograd.Variable(signal_t), torch.autograd.Variable(target_t)
        self.optimizer.zero_grad()
        output = self.net(signal_v)
        loss = self.criterion(output, signal_v)
        loss.backward()
        self.optimizer.step()
        print("loss:", loss.data[0])
    
    def train_legacy(self, x, y, validate=False):
        """
        Parameters:
        validate -- boolean. Set to True to split training data into additional validation set. After each epoch, the validation
          data will be applied to the current model.
        """
        if validate:
            features_train_pre, labels_train, features_val_pre, labels_val = self._split_data(x, y)
        else:
            features_train_pre, labels_train = x, y
            
        lr = 0.001
        n_epochs = 100

        n_features = features_train_pre.shape[1]

        # feature mean substraction and normalization
        self.mean_features = np.mean(features_train_pre, axis=0)
        self.std_features = np.std(features_train_pre, axis=0)
        self.std_features[self.std_features == 0] = 1  # to avoid division by zero
        features_train = (features_train_pre - self.mean_features)/self.std_features

        dataset = torch.utils.data.TensorDataset(torch.Tensor(features_train), torch.Tensor(labels_train))
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.net = Net2(n_features, dr=0.9).cuda(GPU_ID)
        print(self.net)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = torch.nn.BCELoss().cuda(GPU_ID)
        losses_train = np.zeros(n_epochs)

        if validate:
            features_val = (features_val_pre - self.mean_features)/self.std_features
            losses_val = np.zeros(n_epochs)

        print('{:s}: training with {:d} examples'.format(self.name, labels_train.shape[0]))
        for epoch in range(n_epochs):
            sum_train_loss = 0
            for data in trainloader:
                inputs, labels = torch.autograd.Variable(data[0]).cuda(GPU_ID), torch.autograd.Variable(data[1].float()).cuda(GPU_ID)
                optimizer.zero_grad()
                output = self.net(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                sum_train_loss += loss.data[0]
            losses_train[epoch] = sum_train_loss/len(trainloader)
            optimizer.param_groups[0]['lr'] = lr*(0.999**epoch)

            if validate:
                # Validate current model
                features_val_v = torch.autograd.Variable(torch.FloatTensor(features_val)).cuda(GPU_ID)
                labels_val_v = torch.autograd.Variable(torch.FloatTensor(labels_val)).cuda(GPU_ID)
                self.net.eval()
                labels_pred_val = self.net(features_val_v)
                loss = criterion(labels_pred_val, labels_val_v)        
                self.net.train()
                losses_val[epoch] = loss.data[0]
                print('epoch: {:3d} | training loss: {:.4f} | test loss: {:.4f}'.format(epoch, losses_train[epoch], losses_val[epoch]))
            else:
                print('epoch: {:3d} | training loss: {:.4f}'.format(epoch, losses_train[epoch]))

    def score(self, x):
        print('{:s}: scoring {:d} examples'.format(self.name, x.shape[0]))
        features_pp = (x - self.mean_features)/self.std_features
        self.net.eval()
        features_pp_v = torch.autograd.Variable(torch.FloatTensor(features_pp)).cuda(GPU_ID)
        scores_v = self.net(features_pp_v)
        scores_np = scores_v.data.cpu().numpy()
        return scores_np

    def predict(self, x):
        print('{:s}: predicting {:d} examples'.format(self.name, x.shape[0]))
        features_pp = (x - self.mean_features)/self.std_features
        self.net.eval()
        features_pp_v = torch.autograd.Variable(torch.FloatTensor(features_pp)).cuda(GPU_ID)
        scores_v = self.net(features_pp_v)
        scores_np = scores_v.data.cpu().numpy()

        # apply threshold
        y_pred = np.around(scores_np)
        return y_pred

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        some_param = 32
        self.conv1 = nn.Conv3d(1, some_param, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(some_param)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(some_param, some_param*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm3d(some_param*2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(some_param*2, some_param*4, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm3d(some_param*4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(some_param*4, 1, kernel_size=1)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        return x

class Net_bk(nn.Module):
    def __init__(self):
        super().__init__()
        some_param = 32
        self.conv1 = nn.Conv3d(1, some_param, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(some_param)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(some_param, some_param*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(some_param*2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(some_param*2, some_param*4, kernel_size=5)
        self.bn3 = nn.BatchNorm3d(some_param*4)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(some_param*4, 2, kernel_size=1)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        print(x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        print(x.size())
        covfefe
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        return x
    
