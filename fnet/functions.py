import importlib
import json
import os
import pdb
import sys

def load_model(path_model, gpu_ids=0, module='fnet_model'):
    module_fnet_model = importlib.import_module('fnet.' + module)
    if os.path.isdir(path_model):
        path_model = os.path.join(path_model, 'model.p')
    model = module_fnet_model.Model()
    model.load_state(path_model, gpu_ids=gpu_ids)
    return model

def load_model_from_dir(path_model_dir, gpu_ids=0):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    model = fnet.fnet_model.Model()
    model.load_state(path_model_state, gpu_ids=gpu_ids)
    return model
