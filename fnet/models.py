from fnet.fnet_model import Model  # noqa: F401
from fnet.utils.general_utils import str_to_class
from typing import Optional
import json
import os
import pdb  # noqa: F401
import torch


def load_model(
        path_model: str,
        no_optim: bool = False,
        path_options: Optional[str] = None,
):
    """Loaded saved FnetModel.

    Parameters
    ----------
    path_model
        Path to file in which saved model is saved.
    no_optim
        Set to not the model optimizer.
    path_options
        Path to training options json. For legacy saved models where the
        FnetModel class/kwargs are not not included in the model save file.

    Returns
    -------
    FnetModel
        Loaded FnetModel instance.

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
        print('DEBUG: Initializing new model!')
        fnet_model_class = train_options['fnet_model_class']
        fnet_model_kwargs = train_options['fnet_model_kwargs']
        return str_to_class(fnet_model_class)(**fnet_model_kwargs)
    return load_model(path_model, path_options=path_options)
