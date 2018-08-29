import importlib
import inspect
import os
import pdb  # noqa: F401
import re


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


def load_model(path_model, gpu_ids=0, module='fnet_model', no_optim=False):
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
