import os
import shutil
import subprocess


TMP_DIR = '.tmp'


def _init_test():
    path_tmp = os.path.join(os.path.dirname(__file__), TMP_DIR)
    if os.path.exists(path_tmp):
        shutil.rmtree(path_tmp)
    os.makedirs(path_tmp)
    os.chdir(path_tmp)


def _cleanup_test():
    if os.path.basename(os.getcwd()) == TMP_DIR:
        os.chdir(os.path.join(os.getcwd(), os.pardir))
        shutil.rmtree(TMP_DIR)


def test_init():
    _init_test()
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
    _cleanup_test()


def test_train_model_create():
    """Verify that 'fnet train' creates default jsons."""
    _init_test()
    path_create = os.path.join('test_model', 'train_options.json')
    subprocess.run(
        ['fnet', 'train', path_create],
        check=True,
    )
    assert os.path.exists(path_create)
    os.remove(path_create)
    _cleanup_test()


def test_train_model_pred():
    """Verify 'fnet train', 'fnet predict' functionality on an FnetDataset."""
    _init_test()
    path_test_json = os.path.join(
        os.getcwd(), os.pardir, 'data', 'train_options_test.json'
    )
    subprocess.run(
        ['fnet', 'train', path_test_json, '--gpu_ids', '-1'],
        check=True,
    )
    assert os.path.exists('test_model')
    subprocess.run(
        [
            'fnet', 'predict', 'test_model',
            '--dataset', 'tests.data.testdataset',
            '--idx_sel', '0', '3',
            '--gpu_ids', '-1',
        ],
        check=True,
    )
    for fname in ['tifs', 'predictions.csv', 'predict_options.json']:
        assert os.path.exists(os.path.join('predictions', fname))
    _cleanup_test()
