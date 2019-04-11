from typing import Optional
import os
import pathlib
import shutil
import subprocess


def _init_test(path_tmp: Optional[pathlib.Path] = None):
    path_test_dir = pathlib.Path(__file__).parent
    path_data_dir = path_test_dir.parent / 'data'
    if path_tmp is None:
        path_tmp = path_test_dir / '.tmp'
        if path_tmp.exists():
            shutil.rmtree(path_tmp)
        path_tmp.mkdir()
    path_ds_module = path_test_dir / 'data' / 'dummymodule.py'
    shutil.copy(path_ds_module, path_tmp)
    path_tmp_data = path_tmp / 'data'
    path_tmp_data.mkdir()
    for tif in ['EM_low.tif', 'MBP_low.tif']:
        shutil.copy(path_data_dir / tif, path_tmp_data)
    os.chdir(path_tmp)


def test_init(tmp_path: pathlib.Path):
    _init_test(tmp_path)
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


def test_train_model_create(tmp_path: pathlib.Path):
    """Verify that 'fnet train' creates default jsons."""
    _init_test(tmp_path)
    path_create = os.path.join('test_model', 'train_options.json')
    subprocess.run(
        ['fnet', 'train', path_create],
        check=True,
    )
    assert os.path.exists(path_create)


def test_train_model_pred(tmp_path: pathlib.Path):
    """Verify 'fnet train', 'fnet predict' functionality on an FnetDataset."""
    _init_test(tmp_path)
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


def test_train_model_pred_custom(tmp_path: pathlib.Path):
    """Verify 'fnet train', 'fnet predict' functionality on a custom dataset.

    """
    _init_test(tmp_path)
    path_test_json = (
        pathlib.Path(__file__).parent / 'data' / 'train_options_custom.json'
    )
    subprocess.run(
        ['fnet', 'train', str(path_test_json), '--gpu_ids', '-1'],
        check=True,
    )
    assert os.path.exists('test_model')
    subprocess.run(
        [
            'fnet', 'predict', 'test_model',
            '--dataset', 'dummymodule.dummy_custom_dataset',
            '--idx_sel', '2',
            '--gpu_ids', '-1',
        ],
        check=True,
    )
    for fname in ['tifs', 'predictions.csv', 'predict_options.json']:
        assert os.path.exists(os.path.join('predictions', fname))
