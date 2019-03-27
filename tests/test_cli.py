import json
import os
import shutil
import subprocess
import pdb


def test_init():
    path_tmp = os.path.join(os.path.dirname(__file__), '.tmp')
    if os.path.exists(path_tmp):
        shutil.rmtree(path_tmp)
    os.makedirs(path_tmp)
    os.chdir(path_tmp)
    subprocess.run(['fnet init'], shell=True, check=True)
    path_json = os.path.join(
        path_tmp, 'train_options_templates', 'default.json'
    )
    path_script_train = os.path.join(
        path_tmp, 'scripts', 'train_model.py'
    )
    path_script_predict = os.path.join(
        path_tmp, 'scripts', 'predict.py'
    )
    assert os.path.exists(path_json)
    assert os.path.exists(path_script_train)
    assert os.path.exists(path_script_predict)
    shutil.rmtree(path_tmp)


def test_train_model():
    path_tmp = os.path.join(os.path.dirname(__file__), '.tmp')
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

    path_test_json = os.path.join(
        os.getcwd(), os.pardir, 'data', 'train_options_test.json'
    )
    subprocess.run(
        ['fnet', 'train', path_test_json],
        check=True,
    )


if __name__ == '__main__':
    test_train_model()
