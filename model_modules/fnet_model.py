import os
import torch
import importlib
import pdb

class Model(object):
    def __init__(
            self,
            nn_module = 'ttf_v8_nn',
            init_weights = True,
            lr = 0.001,
            criterion_fn = torch.nn.MSELoss, 
            gpu_ids = 0,
    ):
        self.nn_module = nn_module
        self.init_weights = init_weights
        self.lr = lr
        self.criterion_fn = criterion_fn
        self.count_iter = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        
        self.criterion = criterion_fn()
        self._init_model()

    def _init_model(self):
        self.net = importlib.import_module('model_modules.nn_modules.' + self.nn_module).Net()
        if self.init_weights:
            self.net.apply(_weights_init)
        if self.gpu_ids[0] != -1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def __str__(self):
        out_str = '{:s} | iter: {:d}'.format(
            self.nn_module,
            self.count_iter,
        )
        return out_str

    def get_state(self):
        # get nn state
        module = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        module.cpu()
        nn_state = module.state_dict()
        if self.gpu_ids[0] != -1:
            module.cuda(self.gpu_ids[0])
        # get optimizer state
        self.optimizer.state = _set_gpu_recursive(self.optimizer.state, -1)
        optimizer_state = self.optimizer.state_dict()
        self.optimizer.state = _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])
        
        return dict(
            nn_module = self.nn_module,
            nn_state = nn_state,
            optimizer_state = optimizer_state,
            count_iter = self.count_iter,
        )

    def save_state(self, path_save):
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.get_state(), path_save)
        print('model state saved to:', path_save)

    def load_state(self, path_load):
        state_dict = torch.load(path_load)
        print('model state loaded from:', path_load)
        self.nn_module = state_dict['nn_module']
        self._init_model()

        # load nn state
        module = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        module.cpu()
        module.load_state_dict(state_dict['nn_state'])
        if self.gpu_ids[0] != -1:
            module.cuda(self.gpu_ids[0])
        # load optimizer state
        self.optimizer.state = _set_gpu_recursive(self.optimizer.state, -1)
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.optimizer.state = _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])
        
        self.count_iter = state_dict['count_iter']

    def set_lr(self, lr):
        lr_old = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = lr
        print('learning rate: {} => {}'.format(lr_old, lr))

    def do_train_iter(self, signal, target):
        self.net.train()
        if self.gpu_ids[0] != -1:
            signal_v = torch.autograd.Variable(torch.Tensor(signal).cuda(self.gpu_ids[0]))
            target_v = torch.autograd.Variable(torch.Tensor(target).cuda(self.gpu_ids[0]))
        else:
            signal_v = torch.autograd.Variable(torch.Tensor(signal))
            target_v = torch.autograd.Variable(torch.Tensor(target))
        self.optimizer.zero_grad()
        output = self.net(signal_v)
        loss = self.criterion(output, target_v)
        loss.backward()
        self.optimizer.step()
        # print("iter: {:3d} | loss: {:4f}".format(self.meta['count_iter'], loss.data[0]))
        self.count_iter += 1
        return loss.data[0]
    
    def predict(self, signal):
        self.net.eval()
        if self.gpu_ids[0] == -1:
            print('predicting on CPU')
            signal_t = torch.Tensor(signal)
        else:
            signal_t = torch.Tensor(signal).cuda()
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

def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.

    Modified from pytorch_integrated_cell.

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
