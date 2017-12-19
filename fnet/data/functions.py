import os
from fnet.data.czireader import CziReader, get_czi_metadata
from fnet.data.dataset import DataSet
import importlib
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
