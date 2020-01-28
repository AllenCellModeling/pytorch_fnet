from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from .data.testlib import create_tif_data


def _update_json(path_json: Path, **kwargs):
    def helper(some_dict: dict, updates: dict):
        """Recursively updates a dictionary with another."""
        for key, val in updates.items():
            if not isinstance(val, dict):
                some_dict[key] = val
            else:
                helper(some_dict[key], val)

    with path_json.open("r") as fi:
        options = json.load(fi)
    helper(options, kwargs)
    with path_json.open("w") as fo:
        json.dump(options, fo)


@pytest.fixture(scope="module")
def project_dir():
    """Creates a mock user directory in which fnet commands would be used.

    Copies over example tifs to be used as test data and a dummy module
    containing dataset definitions.

    """
    path_pre = Path.cwd()
    path_tmp = Path(tempfile.mkdtemp())
    path_test_dir = Path(__file__).parent
    path_data_dir = path_test_dir.parent.parent / "data"
    Path.mkdir(path_tmp / "data")
    for tif in ["EM_low.tif", "MBP_low.tif"]:
        shutil.copy(path_data_dir / tif, path_tmp / "data")
    shutil.copy(path_test_dir / "data" / "dummymodule.py", path_tmp)
    os.chdir(path_tmp)
    yield path_tmp
    os.chdir(path_pre)


@pytest.mark.usefixtures("project_dir")
def test_init():
    subprocess.run(["fnet init"], shell=True, check=True)
    path_json = os.path.join("train_options_templates", "default.json")
    path_script_train = os.path.join("scripts", "train_model.py")
    path_script_predict = os.path.join("scripts", "predict.py")
    assert os.path.exists(path_json)
    assert os.path.exists(path_script_train)
    assert os.path.exists(path_script_predict)


@pytest.mark.usefixtures("project_dir")
def test_train_model_create():
    """Verify that 'fnet train' creates default jsons."""
    path_create = os.path.join("created", "train_options.json")
    subprocess.run(["fnet", "train", "--json", path_create], check=True)
    assert os.path.exists(path_create)


@pytest.mark.usefixtures("project_dir")
def test_train_model_pred():
    """Verify 'fnet train', 'fnet predict' functionality on an FnetDataset."""
    path_test_json = Path(__file__).parent / "data" / "train_options_test.json"

    subprocess.run(
        ["fnet", "train", "--json", path_test_json, "--gpu_ids", "-1"], check=True
    )
    assert os.path.exists("test_model")
    subprocess.run(
        [
            "fnet",
            "predict",
            "--path_model_dir",
            "test_model",
            "--dataset",
            "dummymodule.dummy_fnet_dataset",
            "--idx_sel",
            "0",
            "3",
            "--gpu_ids",
            "-1",
        ],
        check=True,
    )
    for fname in ["tifs", "predictions.csv", "predict_options.json"]:
        assert os.path.exists(os.path.join("predictions", fname))


@pytest.mark.usefixtures("project_dir")
def test_train_model_pred_custom():
    """Verify 'fnet train', 'fnet predict' functionality on a custom dataset.

    """
    path_test_json = Path(__file__).parent / "data" / "train_options_custom.json"
    subprocess.run(
        ["fnet", "train", "--json", str(path_test_json), "--gpu_ids", "-1"], check=True
    )
    assert os.path.exists("test_model_custom")
    subprocess.run(
        [
            "fnet",
            "predict",
            "--path_model_dir",
            "test_model_custom",
            "--dataset",
            "dummymodule.dummy_custom_dataset",
            "--idx_sel",
            "2",
            "--gpu_ids",
            "-1",
        ],
        check=True,
    )
    for fname in ["tifs", "predictions.csv", "predict_options.json"]:
        assert os.path.exists(os.path.join("predictions", fname))


def train_pred_with_weights(tmp_path):
    shape = (8, 16, 32)
    n_items = 8
    path_ds = create_tif_data(tmp_path, shape=shape, n_items=n_items, weights=True)
    path_train_json = tmp_path / "model" / "train_options.json"
    subprocess.run(
        ["fnet", "train", str(path_train_json), "--gpu_ids", "-1"], check=True
    )
    _update_json(
        path_train_json,
        dataset_train="fnet.data.TiffDataset",
        dataset_train_kwargs={"path_csv": str(path_ds)},
        dataset_val="fnet.data.TiffDataset",
        dataset_val_kwargs={"path_csv": str(path_ds)},
        bpds_kwargs={"patch_shape": [4, 8, 16]},
        n_iter=16,
        interval_save=8,
        fnet_model_kwargs={"nn_class": "tests.data.nn_test.Net"},
    )
    subprocess.run(
        ["fnet", "train", str(path_train_json), "--gpu_ids", "-1"], check=True
    )
