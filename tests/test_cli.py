import os
import shutil
import subprocess


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
