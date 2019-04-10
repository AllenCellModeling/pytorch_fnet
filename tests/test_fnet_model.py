from fnet.fnet_model import Model
from typing import Union
import fnet
import torch


SOME_PARAM_TEST_VAL = 123


class DummyModel(torch.nn.Module):
    def __init__(self, some_param=42):
        super().__init__()
        self.some_param = some_param
        self.network = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def __call__(self, x):
        return self.network(x)


def get_data(device: Union[int, torch.device]) -> tuple:
    if isinstance(device, int):
        device = (torch.device('cuda', device) if device >= 0 else
                  torch.device('cpu'))
    x = torch.rand(1, 1, 8, 16, 16, device=device)
    y = x*2 + 1
    return x, y


def train_new(path_model):
    gpu_id = (1 if torch.cuda.is_available() else -1)
    x, y = get_data(gpu_id)
    model = Model(
        nn_class='test_fnet_model.DummyModel',
        nn_kwargs={'some_param': SOME_PARAM_TEST_VAL},
    )
    model.to_gpu(gpu_id)
    for idx in range(4):
        loss = model.train_on_batch(x, y)
        print(f'loss: {loss:7.5f}')
    model.save(path_model)


def train_more(path_model):
    gpu_id = (0 if torch.cuda.is_available() else -1)
    x, y = get_data(gpu_id)
    model = fnet.models.load_model(path_model)
    for idx in range(2):
        loss = model.train_on_batch(x, y)
        print(f'loss: {loss:7.5f}')
    assert model.count_iter == 6
    assert model.net.some_param == SOME_PARAM_TEST_VAL


def test_resume(tmpdir):
    path_model = tmpdir.mkdir('test_model').join('model.p').strpath
    train_new(path_model)
    train_more(path_model)
