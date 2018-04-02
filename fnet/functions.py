import fnet.fnet_model
import json
import os
import pdb
import sys

def load_dataset_from_dir(path_dir):
    raise NotImplementedError

def load_dataset_from_json(path_json):
    raise NotImplementedError
    
def load_model_from_dir(path_model_dir, gpu_ids=0, model_kwargs = {}):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    model = fnet.fnet_model.Model()
    model.load_state(path_model_state, gpu_ids=gpu_ids, model_kwargs = model_kwargs)
    return model
