import os
from fnet.data.czireader import CziReader, get_czi_metadata
from fnet.data.dataset import DataSet
import importlib
import fnet.data.transforms
import pdb
import pandas as pd
import sys
import json

CHANNEL_TYPES = ('bf', 'dic', 'dna', 'memb', 'struct')

def get_shape_from_metadata(metadata):
    """Return tuple of CZI's dimensions in order (Z, Y, X)."""
    tag_list = 'Metadata.Information.Image'.split('.')
    elements = get_czi_metadata(metadata, tag_list)
    if elements is None:
        return None
    ele_image = elements[0]
    dim_tags = ('SizeZ', 'SizeY', 'SizeX')
    shape = []
    for dim_tag in dim_tags:
        bad_dim = False
        try:
            ele_dim = get_czi_metadata(ele_image, [dim_tag, 'text'])
            shape_dim = int(ele_dim[0])
        except:
            bad_dim = True
        if bad_dim:
            return None
        shape.append(shape_dim)
    return tuple(shape)

def save_dataset_as_json(
        path_train_csv,
        path_test_csv,
        scale_z,
        scale_xy,
        transforms_signal,
        transforms_target,
        path_save,
        name_dataset_module,
):
    dict_ds = dict(
        path_train_csv = path_train_csv,
        path_test_csv = path_test_csv,
        scale_z = scale_z,
        scale_xy = scale_xy,
        transforms_signal = transforms_signal,
        transforms_target = transforms_target,
        name_dataset_module = name_dataset_module,
    )
    with open(path_save, 'w') as fo:
        json.dump(dict_ds, fo)
        
def load_dataset_from_dir(path_model_dir, gpu_ids=0):
    assert os.path.isdir(path_model_dir)
    path_dataset = os.path.join(path_model_dir, 'ds.json')
    dataset = load_dataset_from_json(path_dataset)
    return dataset
    

def load_dataset_from_json(
        path_load,
):
    def get_obj(a):
        if a is None:
            return None
        a_list = a.split('.')
        obj = getattr(sys.modules[__name__], a_list[0])
        for i in range(1, len(a_list)):
            obj = getattr(obj, a_list[i])
        return obj

    with open(path_load, 'r') as fi:
        dict_ds = json.load(fi)
    transforms_signal, transforms_target = None, None
    if dict_ds.get('transforms_signal') is not None:
        transforms_signal = [get_obj(i) for i in dict_ds.get('transforms_signal')]
    if dict_ds.get('transforms_target') is not None:
        transforms_target = [get_obj(i) for i in dict_ds.get('transforms_target')]
    name_dataset_module = dict_ds.get('name_dataset_module', 'fnet.data.dataset')
    transforms = (transforms_signal, transforms_target)
    dataset_module = importlib.import_module(name_dataset_module)
    dataset = dataset_module.DataSet(
        path_train_csv = dict_ds['path_train_csv'],
        path_test_csv = dict_ds['path_test_csv'],
        scale_z = dict_ds['scale_z'],
        scale_xy = dict_ds['scale_xy'],
        transforms=transforms,
    )
    return dataset
