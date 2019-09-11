from typing import Union

import numpy as np
import pytest
import tifffile
import torch

from fnet.fnet_model import Model
import fnet


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


def test_apply_on_single_zstack(tmp_path):
    """Tests the apply_on_single_zstack() method in fnet_model.Model."""
    model = Model(nn_class='test_fnet_model.DummyModel')

    # Test bad inputs
    ar_in = np.random.random(size=(3, 32, 64, 128))
    with pytest.raises(ValueError):
        model.apply_on_single_zstack()
    with pytest.raises(ValueError):
        model.apply_on_single_zstack(ar_in)
    with pytest.raises(ValueError):
        model.apply_on_single_zstack(ar_in[0, 1, ])  # 2d input

    # Test numpy input and file path input
    yhat_ch1 = model.apply_on_single_zstack(ar_in, inputCh=1)
    ar_in = ar_in[1, ]
    path_input_save = tmp_path / 'input_save.tiff'
    tifffile.imsave(str(path_input_save), ar_in, compress=2)
    yhat = model.apply_on_single_zstack(ar_in)
    yhat_file = model.apply_on_single_zstack(filename=path_input_save)
    assert np.issubdtype(yhat.dtype, np.floating)
    assert yhat.shape == ar_in.shape
    assert np.array_equal(yhat, yhat_ch1)
    assert np.array_equal(yhat, yhat_file)

    # Test resized
    factors = (1, .5, .3)
    shape_exp = tuple(round(ar_in.shape[i]*factors[i]) for i in range(3))
    yhat_resized = model.apply_on_single_zstack(ar_in, ResizeRatio=factors)
    assert yhat_resized.shape == shape_exp

    # Test cutoff
    cutoff = 0.1
    yhat_exp = (yhat >= cutoff).astype(np.uint8)*255
    yhat_cutoff = model.apply_on_single_zstack(ar_in, cutoff=cutoff)
    assert np.issubdtype(yhat_cutoff.dtype, np.unsignedinteger)
    assert np.array_equal(yhat_cutoff, yhat_exp)
