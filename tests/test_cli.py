import json
import os
import shutil
import subprocess
import pdb


TMP_DIR = '.tmp'


def test_init():
    path_tmp = os.path.join(os.path.dirname(__file__), TMP_DIR)
    if os.path.exists(path_tmp):
        shutil.rmtree(path_tmp)
    os.makedirs(path_tmp)
    os.chdir(path_tmp)
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
    os.chdir(os.path.join(os.getcwd(), os.pardir))
    shutil.rmtree(TMP_DIR)


def test_train_model():
    path_tmp = os.path.join(os.path.dirname(__file__), TMP_DIR)
    if os.path.exists(path_tmp):
        shutil.rmtree(path_tmp)
    os.makedirs(path_tmp)
    os.chdir(path_tmp)

    # Check that 'fnet train' creates default jsons
    path_create = os.path.join('test_model', 'train_options.json')
    subprocess.run(
        ['fnet', 'train', path_create],
        check=True,
    )
    assert os.path.exists(path_create)
    os.remove(path_create)

    # Train a model using existing json
    path_test_json = os.path.join(
        os.getcwd(), os.pardir, 'data', 'train_options_test.json'
    )
    subprocess.run(
        ['fnet', 'train', path_test_json],
        check=True,
    )
    assert os.path.exists('test_model')
    os.chdir(os.path.join(os.getcwd(), os.pardir))
    shutil.rmtree(TMP_DIR)


if __name__ == '__main__':
    test_init()
    test_train_model()
