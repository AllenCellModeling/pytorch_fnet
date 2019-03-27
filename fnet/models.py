from fnet.fnet_ensemble import FnetEnsemble
from fnet.fnet_model import Model
from fnet.utils.general_utils import str_to_class
from typing import List, Optional, Union
import json
import os
import torch


def load_model(
        path_model: str,
        no_optim: bool = False,
        path_options: Optional[str] = None,
) -> Model:
    """Loaded saved FnetModel.

    Parameters
    ----------
    path_model
        Path to model. If path is a directory, assumes directory contains an
        ensemble of models.
    no_optim
        Set to not the model optimizer.
    path_options
        Path to training options json. For legacy saved models where the
        FnetModel class/kwargs are not not included in the model save file.

    Returns
    -------
    Model or FnetEnsemble
        Loaded model.

    """
    state = torch.load(path_model)
    if 'fnet_model_class' not in state:
        if path_options is not None:
            with open(path_options, 'r') as fi:
                train_options = json.load(fi)
            if 'fnet_model_class' in train_options:
                state['fnet_model_class'] = train_options['fnet_model_class']
                state['fnet_model_kwargs'] = train_options['fnet_model_kwargs']
    fnet_model_class = state.get('fnet_model_class', 'fnet.models.Model')
    fnet_model_kwargs = state.get('fnet_model_kwargs', {})
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
        with open(path_options, 'r') as fi:
            train_options = json.load(fi)
        print('Initializing new model!')
        fnet_model_class = train_options['fnet_model_class']
        fnet_model_kwargs = train_options['fnet_model_kwargs']
        return str_to_class(fnet_model_class)(**fnet_model_kwargs)
    return load_model(path_model, path_options=path_options)


def create_ensemble(
        paths_model: Union[str, List[str]],
        path_save_dir: str,
) -> None:
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
        paths_model = paths_model.split(' ')
    paths_member = []
    for path_model in paths_model:
        path_model = os.path.abspath(path_model)
        if os.path.isdir(path_model):
            path_member = os.path.join(path_model, 'model.p')
            if os.path.exists(path_member):
                paths_member.append(path_member)
                continue
            paths_member.extend(sorted(
                [
                    p.path for p in os.scandir(path_model)
                    if p.path.endswith('.p')
                ]
            ))
        else:
            paths_member.append(path_model)
    path_save = os.path.join(path_save_dir, 'model.p')
    ensemble = FnetEnsemble(paths_model=paths_member)
    ensemble.save(path_save)
