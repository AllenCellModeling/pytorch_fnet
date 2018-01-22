import fnet.fnet_model
import json
import os
import pdb
import sys

def load_dataset_from_dir(path_dir):
    raise NotImplementedError

def load_dataset_from_json(path_json):
    raise NotImplementedError
    
def load_model_from_dir(path_model_dir, gpu_ids=0):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    model = fnet.fnet_model.Model(
        gpu_ids=gpu_ids,
    )
    model.load_state(path_model_state)
    return model
