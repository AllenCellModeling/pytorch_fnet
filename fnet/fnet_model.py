from fnet.utils.general_utils import get_args, retry_if_oserror, str_to_class
from fnet.utils.model_utils import move_optim
from typing import Union, Iterator
import math
import os
import pdb
import torch


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 


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


class Model:
    """Class that encompasses a pytorch network and its optimizer.

    """

    def __init__(
            self,
            betas=(0.5, 0.999),
            criterion_class='torch.nn.MSELoss',
            init_weights=True,
            lr=0.001,
            nn_class='fnet.nn_modules.fnet_nn_3d.Net',
            nn_kwargs={},
            nn_module=None,
            scheduler=None,
            weight_decay=0,
            gpu_ids=-1,
    ):
        self.betas = betas
        self.criterion = str_to_class(criterion_class)()
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.init_weights = init_weights
        self.lr = lr
        self.nn_class = nn_class
        self.nn_kwargs = nn_kwargs
        self.scheduler = scheduler
        self.weight_decay = weight_decay

        # *** Legacy support ***
        # self.nn_module might be specified in legacy saves.
        # If so, override self.nn_class
        if nn_module is not None:
            self.nn_class = nn_module + '.Net'
        
        self.count_iter = 0
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self._init_model()
        self.fnet_model_kwargs, self.fnet_model_posargs = get_args()
        self.fnet_model_kwargs.pop('self')

    def _init_model(self):
        self.net = str_to_class(self.nn_class)(
            **self.nn_kwargs
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
        out_str = [
            f'*** {self.__class__.__name__} ***',
            f'{self.nn_class}(**{self.nn_kwargs})',
            f'iter: {self.count_iter}',
            f'gpu: {self.gpu_ids}',
        ]
        return os.linesep.join(out_str)

    def get_state(self):
        return {
            'fnet_model_class': (self.__module__ + '.' +
                                 self.__class__.__qualname__),
            'fnet_model_kwargs': self.fnet_model_kwargs,
            'fnet_model_posargs': self.fnet_model_posargs,
            'nn_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'count_iter': self.count_iter,
        }

    def to_gpu(self, gpu_ids: Union[int, list, ]):
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids
        self.device = (
            torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else
            torch.device('cpu')
        )
        self.net.to(self.device)
        if self.optimizer is not None:
            move_optim(self.optimizer, self.device)

    def save(self, path_save: str):
        """Saves model to disk.

        Parameters
        ----------
        path_save
            Filename to which model is saved.

        """
        assert not os.path.isdir(path_save)
        curr_gpu_ids = self.gpu_ids
        self.to_gpu(-1)
        retry_if_oserror(torch.save)(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def load_state(self, state: dict, no_optim: bool = False):
        self.count_iter = state['count_iter']
        self.net.load_state_dict(state['nn_state'])
        if no_optim:
            self.optimizer = None
            return
        self.optimizer.load_state_dict(state['optimizer_state'])

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

    def test_on_batch(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> float:
        """Test model on a batch of inputs and targets.

        Parameters
        ----------
        x
            Batched input.
        y
            Batched target.

        Returns
        -------
        float
            Loss as evaluated by self.criterion.

        """
        if len(self.gpu_ids) > 1:
            network = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
        else:
            network = self.net
        network.eval()
        x = x.to(dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y_hat = network(x).cpu()
        loss = self.criterion(y_hat, y)
        network.train()
        return loss.item()

    def test_on_iterator(
            self,
            iterator: Iterator,
            **kwargs: dict,
    ) -> float:
        """Test model on iterator which has items to be passed to
        test_on_batch.

        Parameters
        ----------
        iterator
            Iterator that generates items to be passed to test_on_batch.
        kwargs
            Additional keyword arguments to be passed to test_on_batch.

        Returns
        -------
        float
            Mean loss for items in iterable.

        """
        loss_sum = 0
        for item in iterator:
            loss_sum += self.test_on_batch(*item, **kwargs)
        return loss_sum/len(iterator)
