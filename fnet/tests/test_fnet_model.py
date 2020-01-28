from typing import Union

import numpy as np
import numpy.testing as npt
import pytest
import tifffile
import torch

import fnet
from fnet.fnet_model import Model

SOME_PARAM_TEST_VAL = 123


def get_data(device: Union[int, torch.device]) -> tuple:
    if isinstance(device, int):
        device = torch.device("cuda", device) if device >= 0 else torch.device("cpu")
    x = torch.rand(1, 1, 8, 16, 16, device=device)
    y = x * 2 + 1
    return x, y


def train_new(path_model):
    gpu_id = 1 if torch.cuda.is_available() else -1
    x, y = get_data(gpu_id)
    model = Model(
        nn_class="fnet.nn_modules.dummy.DummyModel",
        nn_kwargs={"some_param": SOME_PARAM_TEST_VAL},
    )
    model.to_gpu(gpu_id)
    for idx in range(4):
        _ = model.train_on_batch(x, y)
    model.save(path_model)


def train_more(path_model):
    gpu_id = 0 if torch.cuda.is_available() else -1
    x, y = get_data(gpu_id)
    model = fnet.models.load_model(path_model)
    for idx in range(2):
        _ = model.train_on_batch(x, y)
    assert model.count_iter == 6
    assert model.net.some_param == SOME_PARAM_TEST_VAL


def test_resume(tmpdir):
    path_model = tmpdir.mkdir("test_model").join("model.p").strpath
    train_new(path_model)
    train_more(path_model)


def test_apply_on_single_zstack(tmp_path):
    """Tests the apply_on_single_zstack() method in fnet_model.Model."""
    model = Model(nn_class="fnet.nn_modules.dummy.DummyModel")

    # Test bad inputs
    ar_in = np.random.random(size=(3, 32, 64, 128))
    with pytest.raises(ValueError):
        model.apply_on_single_zstack()
    with pytest.raises(ValueError):
        model.apply_on_single_zstack(ar_in)
    with pytest.raises(ValueError):
        model.apply_on_single_zstack(ar_in[0, 1])  # 2d input

    # Test numpy input and file path input
    yhat_ch1 = model.apply_on_single_zstack(ar_in, inputCh=1)
    ar_in = ar_in[1,]
    path_input_save = tmp_path / "input_save.tiff"
    tifffile.imsave(str(path_input_save), ar_in, compress=2)
    yhat = model.apply_on_single_zstack(ar_in)
    yhat_file = model.apply_on_single_zstack(filename=path_input_save)
    assert np.issubdtype(yhat.dtype, np.floating)
    assert yhat.shape == ar_in.shape
    assert np.array_equal(yhat, yhat_ch1)
    assert np.array_equal(yhat, yhat_file)

    # Test resized
    factors = (1, 0.5, 0.3)
    shape_exp = tuple(round(ar_in.shape[i] * factors[i]) for i in range(3))
    yhat_resized = model.apply_on_single_zstack(ar_in, ResizeRatio=factors)
    assert yhat_resized.shape == shape_exp

    # Test cutoff
    cutoff = 0.1
    yhat_exp = (yhat >= cutoff).astype(np.uint8) * 255
    yhat_cutoff = model.apply_on_single_zstack(ar_in, cutoff=cutoff)
    assert np.issubdtype(yhat_cutoff.dtype, np.unsignedinteger)
    assert np.array_equal(yhat_cutoff, yhat_exp)


def test_train_on_batch():
    model = Model(nn_class="fnet.tests.data.nn_test.Net", lr=0.01)
    shape_item = (1, 2, 4, 8)
    batch_size = 9
    shape_batch = (batch_size,) + shape_item
    x_batch = torch.rand(shape_batch)
    y_batch = x_batch * 0.666 + 0.42
    cost_prev = float("inf")
    for _ in range(8):
        cost = model.train_on_batch(x_batch, y_batch)
        assert cost < cost_prev
        cost_prev = cost

    # Test target weight maps
    model = Model(nn_class="fnet.tests.data.nn_test.Net", lr=0.0)  # disable learning
    cost_norm = model.train_on_batch(x_batch, y_batch)
    # Test uniform weight map
    weight_map_batch = (torch.ones(shape_item) / np.prod(shape_item)).expand(
        shape_batch
    )
    cost_weighted = model.train_on_batch(x_batch, y_batch, weight_map_batch)
    npt.assert_approx_equal(cost_weighted, cost_norm, significant=6)
    # Test all-zero weight map
    cost_weighted = model.train_on_batch(x_batch, y_batch, torch.zeros(x_batch.size()))
    npt.assert_approx_equal(cost_weighted, 0.0)
    # Random weights with first and last examples having zero weight
    weight_map_batch = torch.rand(shape_batch)
    weight_map_batch[[0, -1]] = 0.0
    cost_weighted = model.train_on_batch(x_batch, y_batch, weight_map_batch)
    cost_exp = (
        model.train_on_batch(x_batch[1:-1], y_batch[1:-1], weight_map_batch[1:-1])
        * (batch_size - 2)
        / batch_size  # account for change in batch size
    )
    npt.assert_approx_equal(cost_weighted, cost_exp)


def test_test_on_batch():
    model = Model(nn_class="fnet.tests.data.nn_test.Net", lr=0.01)
    shape_item = (1, 2, 4, 8)
    batch_size = 1
    shape_batch = (batch_size,) + shape_item
    x_batch = torch.rand(shape_batch)
    y_batch = x_batch * 0.666 + 0.42

    # Model weights should remain the same so loss should not change
    loss_0 = model.test_on_batch(x_batch, y_batch)
    loss_1 = model.test_on_batch(x_batch, y_batch)
    npt.assert_approx_equal(loss_1 - loss_0, 0.0)

    # Loss should remain the same with uniform weight map
    loss_weight_uniform = model.test_on_batch(
        x_batch, y_batch, torch.ones(shape_batch) / np.prod(shape_item)
    )
    npt.assert_almost_equal(loss_weight_uniform - loss_0, 0.0)

    # Loss should be zero with all-zero weight map
    loss_weight_zero = model.test_on_batch(x_batch, y_batch, torch.zeros(shape_batch))
    npt.assert_almost_equal(loss_weight_zero, 0.0)
