from fnet.fnet_model import Model  # noqa: F401
from fnet.utils.general_utils import str_to_class
import json
import os
import pdb  # noqa: F401
import torch


def load_model(path_model: str, no_optim: bool = False):
    state = torch.load(path_model)
    if 'fnet_model_class' not in state:
        print('Using default FnetModel: fnet.models.Model')
    fnet_model_class = state.get('fnet_model_class', 'fnet.models.Model')
    fnet_model_kwargs = state.get('fnet_model_kwargs', {})
    model = str_to_class(fnet_model_class)(**fnet_model_kwargs)
    model.load_state(state, no_optim)
    return model


def load_model_from_dir(path_load_dir: str, no_optim: bool = False):
    path_options = os.path.join(path_load_dir, 'train_options.json')
    with open(path_options, 'r') as fi:
        train_options = json.load(fi)
    fnet_model_kwargs = train_options['fnet_model_kwargs']
    fnet_model_class = train_options['fnet_model_class']
    model = str_to_class(fnet_model_class)(**fnet_model_kwargs)
    path_saved_state = os.path.join(path_load_dir, 'model.p')
    if os.path.exists(path_saved_state):
        state = torch.load(path_saved_state)
        model.load_state(state, no_optim)
    return model
