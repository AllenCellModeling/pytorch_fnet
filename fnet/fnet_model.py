import os
import torch
import importlib
import pdb

class Model(object):
    def __init__(
            self,
            nn_module = None,
            init_weights = True,
            lr = 0.001,
            weight_decay = 0,
            adamw_decay = 0,
            criterion_fn = torch.nn.MSELoss, 
            nn_kwargs={},
            gpu_ids = -1,
    ):
        self.nn_module = nn_module
        self.nn_kwargs = nn_kwargs
        self.init_weights = init_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion_fn = criterion_fn
        self.count_iter = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.weight_init_key = 'net_recurse.sub_2conv_more.net.0.weight' if self.nn_module == 'fnet_nn_3d_uncertainty' else 'net_recurse.sub_2conv_more.conv1.weight'
        self.adamw_decay = adamw_decay
        
        if criterion_fn is not None:
            self.criterion = criterion_fn()
            
        self._init_model(nn_kwargs=self.nn_kwargs)

    def _init_model(self, nn_kwargs={}):
        if self.nn_module is None:
            self.net = None
            return
        self.net = importlib.import_module('fnet.nn_modules.' + self.nn_module).Net(**nn_kwargs)
        if self.init_weights:
            self.net.apply(_weights_init)
        if self.gpu_ids[0] >= 0:
            self.net.cuda(self.gpu_ids[0])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.5, 0.999))

    def __str__(self):
        out_str = '{:s} | {:s} | iter: {:d}'.format(
            self.nn_module,
            str(self.nn_kwargs),
            self.count_iter,
        )
        return out_str

    def get_state(self):
        return dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            count_iter = self.count_iter,
        )

    def to_gpu(self, gpu_ids):
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        if gpu_ids[0] >= 0:
            self.net.cuda(gpu_ids[0])
        else:
            self.net.cpu()
        _set_gpu_recursive(self.optimizer.state, gpu_ids[0])  # this may not work in the future
        self.gpu_ids = gpu_ids

    def save_state(self, path_save):
        curr_gpu_ids = self.gpu_ids
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_gpu(-1)
        torch.save(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def load_state(self, path_load, gpu_ids=-1):
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(nn_kwargs=self.nn_kwargs)
        self.net.load_state_dict(state_dict['nn_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.count_iter = state_dict['count_iter']
        self.to_gpu(gpu_ids)
        
    def load_weights(self, path_load, gpu_ids=-1):
        source_model = torch.load(path_load)
        source_in_channels = source_model.get('nn_kwargs', {}).get('in_channels', 1)
        source_state_dict = source_model['nn_state']
        state_dict = self.net.state_dict()
        weights = state_dict[self.weight_init_key].cpu()[slice(None),slice(0, self.net.in_channels - source_in_channels)]
        source_weights = source_state_dict[self.weight_init_key]
        source_state_dict[self.weight_init_key] = torch.cat((weights, source_weights), 1)
        self.net.load_state_dict(source_state_dict)

    def do_train_iter(self, signal, target):
        self.net.train()
        if self.gpu_ids[0] >= 0:
            signal_v = torch.autograd.Variable(signal.cuda(self.gpu_ids[0]))
            target_v = torch.autograd.Variable(target.cuda(self.gpu_ids[0]))
        else:
            signal_v = torch.autograd.Variable(signal)
            target_v = torch.autograd.Variable(target)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        self.optimizer.zero_grad()
        signal_v.requires_grad = True
        
        output = module(signal_v)
        loss = self.criterion(output, target_v)
        loss.backward()
        if self.adamw_decay > 0:
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-self.adamw_decay * self.lr, param.data)
        self.optimizer.step()
        self.count_iter += 1
        
        return loss.data[0]
    
    def predict(self, signal):
        if self.gpu_ids[0] >= 0:
            signal = signal.cuda(self.gpu_ids[0])
        else:
            print('predicting on CPU')
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        signal_v = torch.autograd.Variable(signal, volatile=True)
        module.eval()
        return module(signal_v).data.cpu()

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 

def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.
    Modified from pytorch_integrated_cell.
    Parameters:
    var - (dict) keys are either Tensors or dicts
    gpu_id - (int) GPU onto which to move the Tensors
    """
    for key in var:
        if isinstance(var[key], dict):
            _set_gpu_recursive(var[key], gpu_id)
        elif torch.is_tensor(var[key]):
            if gpu_id == -1:
                var[key] = var[key].cpu()
            else:
                var[key] = var[key].cuda(gpu_id)
