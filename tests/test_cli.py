from typing import Optional
import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest


@pytest.fixture(scope='module')
def project_dir():
    """Creates a mock user directory in which fnet commands would be used.

    Copies over example tifs to be used as test data and a dummy module
    containing dataset definitions.

    """
    path_pre = pathlib.Path.cwd()
    path_tmp = pathlib.Path(tempfile.mkdtemp())
    path_test_dir = pathlib.Path(__file__).parent
    path_data_dir = path_test_dir.parent / 'data'
    pathlib.Path.mkdir(path_tmp / 'data')
    for tif in ['EM_low.tif', 'MBP_low.tif']:
        shutil.copy(path_data_dir / tif, path_tmp / 'data')
    shutil.copy(path_test_dir / 'data' / 'dummymodule.py', path_tmp)
    os.chdir(path_tmp)
    yield path_tmp
    os.chdir(path_pre)


@pytest.mark.usefixtures('project_dir')
def test_init():
    subprocess.run(['fnet init'], shell=True, check=True)
    path_json = os.path.join(
        'train_options_templates', 'default.json'
    )
    path_script_train = os.path.join(
        'scripts', 'train_model.py'
    )
    path_script_predict = os.path.join(
        'scripts', 'predict.py'
    )
    assert os.path.exists(path_json)
    assert os.path.exists(path_script_train)
    assert os.path.exists(path_script_predict)


@pytest.mark.usefixtures('project_dir')
def test_train_model_create():
    """Verify that 'fnet train' creates default jsons."""
    path_create = os.path.join('created', 'train_options.json')
    subprocess.run(
        ['fnet', 'train', path_create],
        check=True,
    )
    assert os.path.exists(path_create)


@pytest.mark.usefixtures('project_dir')
def test_train_model_pred():
    """Verify 'fnet train', 'fnet predict' functionality on an FnetDataset."""
    path_test_json = (
        pathlib.Path(__file__).parent / 'data' / 'train_options_test.json'
    )
    subprocess.run(
        ['fnet', 'train', path_test_json, '--gpu_ids', '-1'],
        check=True,
    )
    assert os.path.exists('test_model')
    subprocess.run(
        [
            'fnet', 'predict', 'test_model',
            '--dataset', 'dummymodule.dummy_fnet_dataset',
            '--idx_sel', '0', '3',
            '--gpu_ids', '-1',
        ],
        check=True,
    )
    for fname in ['tifs', 'predictions.csv', 'predict_options.json']:
        assert os.path.exists(os.path.join('predictions', fname))


@pytest.mark.usefixtures('project_dir')
def test_train_model_pred_custom():
    """Verify 'fnet train', 'fnet predict' functionality on a custom dataset.

    """
    path_test_json = (
        pathlib.Path(__file__).parent / 'data' / 'train_options_custom.json'
    )
    subprocess.run(
        ['fnet', 'train', str(path_test_json), '--gpu_ids', '-1'],
        check=True,
    )
    assert os.path.exists('test_model_custom')
    subprocess.run(
        [
            'fnet', 'predict', 'test_model_custom',
            '--dataset', 'dummymodule.dummy_custom_dataset',
            '--idx_sel', '2',
            '--gpu_ids', '-1',
        ],
        check=True,
    )
    for fname in ['tifs', 'predictions.csv', 'predict_options.json']:
        assert os.path.exists(os.path.join('predictions', fname))
