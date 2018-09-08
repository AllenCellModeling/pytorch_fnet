from typing import Optional
import importlib
import inspect
import json
import logging
import os
import pdb  # noqa: F401
import re


def str_to_class(string: str):
    """Return class from string representation."""
    idx_dot = string.rfind('.')
    if idx_dot < 0:
        module_str = 'fnet.nn_modules'
        class_str = string
    else:
        module_str = string[:idx_dot]
        class_str = string[idx_dot + 1:]
    module = importlib.import_module(module_str)
    return getattr(module, class_str)


def to_objects(slist):
    """Get a list of objects from list of object __repr__s."""
    if slist is None:
        return list()
    olist = list()
    for s in slist:
        if not isinstance(s, str):
            if s is None:
                continue
            olist.append(s)
            continue
        if s.lower() == 'none':
            continue
        s_split = s.split('.')
        for idx_part, part in enumerate(s_split):
            if not part.isidentifier():
                break
        importee = '.'.join(s_split[:idx_part])
        so = '.'.join(s_split[idx_part:])
        if len(importee) > 0:
            module = importlib.import_module(importee)  # noqa: F841
            so = 'module.' + so
        olist.append(eval(so))
    return olist


def load_model(
        path_save_dir: str, no_optim=False,
        logger: Optional[logging.Logger] = None,
):
    printl = print if logger is None else logger.info
    path_options = os.path.join(path_save_dir, 'train_options.json')
    with open(path_options, 'r') as fi:
        train_options = json.load(fi)
    fnet_model_kwargs = train_options['fnet_model_kwargs']
    fnet_model_class = train_options['fnet_model_class']
    model = str_to_class(fnet_model_class)(**fnet_model_kwargs)
    path_saved_state = os.path.join(path_save_dir, 'model.p')
    if os.path.exists(path_saved_state):
        model.load_state(path_saved_state, no_optim)
        printl('Loaded model state from:', path_saved_state)
    return model


def load_model_archive(path_model, gpu_ids=0, module='fnet_model', no_optim=False):
    """Load a model from a path to a model.

    Args:
      path_model: The path to the model. Can be a directory, a pickle file, or
        a directory with a checkpoint specified in this format:
        "directory:checkpoint".
      gpu_ids: The GPU ID on which to load the model.
      module: The module name of the with fnet_model wrapper class.
    """
    module_fnet_model = importlib.import_module('fnet.' + module)
    if ':' in path_model:
        idx_colon = path_model.rfind(':')
        checkpoint = int(path_model[idx_colon + 1:])
        path_model_root = path_model[:idx_colon]
        reo = re.compile(r'model_(\d+)\.p')
        for cp in os.listdir(os.path.join(path_model_root, 'checkpoints')):
            match = reo.match(cp)
            if match is not None and int(match.groups()[0]) == checkpoint:
                path_model = os.path.join(path_model_root, 'checkpoints', cp)
                break
    elif os.path.isdir(path_model):
        path_model = os.path.join(path_model, 'model.p')
    if 'load_path' in inspect.getfullargspec(module_fnet_model.Model).args:
        # For when Model has 'load_path' as a ctor parameter
        return module_fnet_model.Model(load_path=path_model, gpu_ids=gpu_ids)
    model = module_fnet_model.Model()
    model.load_state(path_model, gpu_ids=gpu_ids, no_optim=no_optim)
    return model
