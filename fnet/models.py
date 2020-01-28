from typing import List, Optional, Union
import json
import logging
import os

import torch

from fnet.fnet_ensemble import FnetEnsemble
from fnet.fnet_model import Model
from fnet.utils.general_utils import str_to_class


logger = logging.getLogger(__name__)


def _find_model_checkpoint(path_model_dir: str, checkpoint: str):
    """Finds path to a specific model checkpoint.

    Parameters
    ----------
    path_model_dir
        Path to model as a directory.
    checkpoint
        String that identifies a model checkpoint

    Returns
    -------
    str
        Path to saved model file.

    """
    path_cp_dir = os.path.join(path_model_dir, "checkpoints")
    if not os.path.exists(path_cp_dir):
        raise ValueError(f"Model ({path_cp_dir} has no checkpoints)")
    paths_cp = sorted(
        [p.path for p in os.scandir(path_cp_dir) if p.path.endswith(".p")]
    )
    for path_cp in paths_cp:
        if checkpoint in os.path.basename(path_cp):
            return path_cp
    raise ValueError(f"Model checkpoint not found: {checkpoint}")


def load_model(
    path_model: str,
    no_optim: bool = False,
    checkpoint: Optional[str] = None,
    path_options: Optional[str] = None,
) -> Model:
    """Loaded saved FnetModel.

    Parameters
    ----------
    path_model
        Path to model as a directory or .p file.
    no_optim
        Set to not the model optimizer.
    checkpoint
        Optional string that identifies a model checkpoint
    path_options
        Path to training options json. For legacy saved models where the
        FnetModel class/kwargs are not not included in the model save file.

    Returns
    -------
    Model
        Loaded model.

    """
    if not os.path.exists(path_model):
        raise ValueError(f"Model path does not exist: {path_model}")
    if os.path.isdir(path_model):
        if checkpoint is None:
            path_model = os.path.join(path_model, "model.p")
            if not os.path.exists(path_model):
                raise ValueError(f"Default model not found: {path_model}")
        if checkpoint is not None:
            path_model = _find_model_checkpoint(path_model, checkpoint)
    state = torch.load(path_model)
    if "fnet_model_class" not in state:
        if path_options is not None:
            with open(path_options, "r") as fi:
                train_options = json.load(fi)
            if "fnet_model_class" in train_options:
                state["fnet_model_class"] = train_options["fnet_model_class"]
                state["fnet_model_kwargs"] = train_options["fnet_model_kwargs"]
    fnet_model_class = state.get("fnet_model_class", "fnet.models.Model")
    fnet_model_kwargs = state.get("fnet_model_kwargs", {})
    model = str_to_class(fnet_model_class)(**fnet_model_kwargs)
    model.load_state(state, no_optim)
    return model


def load_or_init_model(path_model: str, path_options: str):
    """Loaded saved model if it exists otherwise inititialize new model.

    Parameters
    ----------
    path_model
        Path to saved model.
    path_options
        Path to json where model training options are saved.

    Returns
    -------
    FnetModel
        Loaded or new FnetModel instance.

    """
    if not os.path.exists(path_model):
        with open(path_options, "r") as fi:
            train_options = json.load(fi)
        logger.info("Initializing new model!")
        fnet_model_class = train_options["fnet_model_class"]
        fnet_model_kwargs = train_options["fnet_model_kwargs"]
        return str_to_class(fnet_model_class)(**fnet_model_kwargs)
    return load_model(path_model, path_options=path_options)


def create_ensemble(paths_model: Union[str, List[str]], path_save_dir: str) -> None:
    """Create and save an ensemble model.

    Parameters
    ----------
    paths_model
        Paths to models or model directories. Paths can be specified as items
        in list or as a string with paths separated by spaces. Any model
        specified as a directory assumed to be at 'directory/model.p'.
    path_save_dir
        Model save path directory. Model will be saved at in path_save_dir as
        'model.p'.

    """
    if isinstance(paths_model, str):
        paths_model = paths_model.split(" ")
    paths_member = []
    for path_model in paths_model:
        path_model = os.path.abspath(path_model)
        if os.path.isdir(path_model):
            path_member = os.path.join(path_model, "model.p")
            if os.path.exists(path_member):
                paths_member.append(path_member)
                continue
            paths_member.extend(
                sorted(
                    [p.path for p in os.scandir(path_model) if p.path.endswith(".p")]
                )
            )
        else:
            paths_member.append(path_model)
    path_save = os.path.join(path_save_dir, "model.p")
    ensemble = FnetEnsemble(paths_model=paths_member)
    ensemble.save(path_save)
