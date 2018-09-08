import fnet.functions
import importlib
import math
import os
import pdb
import torch


class Model:
    def __init__(
            self,
            nn_module='fnet.nn_modules.fnet_nn_3d',
            init_weights = True,
            lr = 0.001,
            criterion_fn = torch.nn.MSELoss, 
            nn_kwargs={},
            gpu_ids = -1,
            weight_decay=0,
            betas=(0.5, 0.999),
            scheduler=None,
    ):
        self.nn_module = nn_module
        self.nn_kwargs = nn_kwargs
        self.init_weights = init_weights
        self.lr = lr
        self.criterion_fn = criterion_fn
        self.count_iter = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.weight_decay = weight_decay
        self.betas = betas
        self.scheduler = scheduler
        
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        
        self.criterion = criterion_fn()
        self._init_model(nn_kwargs=self.nn_kwargs)


    def _init_model(self, nn_kwargs={}):
        self.net = fnet.functions.str_to_class(self.nn_module + '.Net')(
            **nn_kwargs
        )
        if self.init_weights:
            self.net.apply(_weights_init)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(
            get_per_param_options(
                self.net, wd=self.weight_decay
            ),
            lr=self.lr,
            betas=self.betas,
        )

        if self.scheduler is not None:
            if self.scheduler[0] == 'snapshot':
                period = self.scheduler[1]
                foo = lambda x: 0.5 + 0.5*math.cos(math.pi*(x % period)/period)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, foo
                )
            elif self.scheduler[0] == 'step':
                step_size = self.scheduler[1]
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size
                )
            else:
                raise NotImplementedError


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
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.net.to(self.device)
        if self.optimizer is not None:
            _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])  # this may not work in the future

    def save_state(self, path_save):
        curr_gpu_ids = self.gpu_ids
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_gpu(-1)
        torch.save(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def load_state(self, path_load, gpu_ids=-1, no_optim=False):
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(nn_kwargs=self.nn_kwargs)
        self.net.load_state_dict(state_dict['nn_state'])
        if no_optim:
            self.optimizer = None
        else:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.count_iter = state_dict['count_iter']
        self.to_gpu(gpu_ids)

    def do_train_iter(self, signal, target):
        if self.scheduler is not None:
            self.scheduler.step()
        self.net.train()
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        self.optimizer.zero_grad()
        output = module(signal)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.count_iter += 1
        return loss.item()
    
    def predict(self, signal):
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        module.eval()
        with torch.no_grad():
            prediction = module(signal).cpu()
        return prediction

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


def get_per_param_options(module, wd):
    """Returns list of per parameter group options.

    Applies the specified weight decay (wd) to parameters except parameters
    within batch norm layers and bias parameters.
    """
    if wd == 0:
        return module.parameters()
    with_decay = list()
    without_decay = list()
    for idx_m, (name_m, module_sub) in enumerate(module.named_modules()):
        if len(list(module_sub.named_children())) > 0:
            continue  # Skip "container" modules
        if isinstance(module_sub, torch.nn.modules.batchnorm._BatchNorm):
            for param in module_sub.parameters():
                without_decay.append(param)
            continue
        for name_param, param in module_sub.named_parameters():
            if 'weight' in name_param:
                with_decay.append(param)
            elif 'bias' in name_param:
                without_decay.append(param)
    # Check that no parameters were missed or duplicated
    n_param_module = len(list(module.parameters()))
    n_param_lists = len(with_decay) + len(without_decay)
    n_elem_module = sum([p.numel() for p in module.parameters()])
    n_elem_lists = sum([p.numel() for p in (with_decay + without_decay)])
    assert n_param_module == n_param_lists
    assert n_elem_module == n_elem_lists
    per_param_options = [
        {
            'params': with_decay,
            'weight_decay': wd,
        },
        {
            'params': without_decay,
            'weight_decay': 0.0,
        },
    ]
    return per_param_options
