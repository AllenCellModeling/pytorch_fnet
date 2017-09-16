import pickle
import os
from util.data.czireader import CziReader
from util.data.dataset1 import DataSet as DataSet1
from util.data.dataset2 import DataSet2
from util.data.dataset import DataSet
import pdb
import pandas as pd

CHANNEL_TYPES = ('bf', 'dic', 'dna', 'memb', 'struct')

def get_czi_metadata(element, tag_list):
    """
    element - (xml.etree.ElementTree.Element)
    tag_list - list of strings
    """
    if len(tag_list) == 0:
        return None
    if len(tag_list) == 1:
        if tag_list[0] == 'attrib':
            return [element.attrib]
        if tag_list[0] == 'text':
            return [element.text]
    values = []
    for sub_ele in element:
        if sub_ele.tag == tag_list[0]:
            if len(tag_list) == 1:
                values.extend([sub_ele])
            else:
                retval = get_czi_metadata(sub_ele, tag_list[1:])
                if retval is not None:
                    values.extend(retval)
    if len(values) == 0:
        return None
    return values

def _shuffle_split_df(df_all, train_split, no_shuffle=False):
    if not no_shuffle:
        df_all_shuf = df_all.sample(frac=1).reset_index(drop=True)
    else:
        df_all_shuf = df_all
    if train_split == 0:
        df_test = df_all_shuf
        df_train = df_all_shuf[0:0]  # empty DataFrame but with columns intact
    else:
        if isinstance(train_split, int):
            idx_split = train_split
        elif isinstance(train_split, float):
            idx_split = round(len(df_all_shuf)*self._percent_test)
        else:
            raise AttributeError
        df_train = df_all_shuf[:idx_split]
        df_test = df_all_shuf[idx_split:]
    return df_train, df_test

def _get_shape_from_metadata(metadata):
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

def _get_chan_idx(channel_nickname, channel_info_dicts):
    retval = None
    for info_dict in channel_info_dicts:
        name = info_dict.get('Name')
        name_match = False
        if channel_nickname == 'bf':
            if name.lower().endswith('brightfield'):
                name_match = True
        elif channel_nickname == 'dic':
            if name.endswith('DIC'):
                name_match = True
        elif channel_nickname == 'dna':
            raise NotImplementedError
        elif channel_nickname == 'memb':
            if name == 'CMDRP':
                name_match = True
        elif channel_nickname == 'struct':
            if name == 'EGFP':
                name_match = True
        else:
            raise NotImplementedError
        if name_match:
            id_val = info_dict.get('Id')
            try:
                chan_idx = int(id_val.split(':')[-1])
                retval = chan_idx
            except:
                break
    return retval

def create_dataset(
        path_source,
        name_signal,
        name_target,
        *args, **kwargs
 ):
    assert (name_signal in CHANNEL_TYPES) and (name_target in CHANNEL_TYPES)
    if os.path.isdir(path_source):
        return create_dataset_from_dir(
            path_dir=path_source,
            name_signal=name_signal,
            name_target=name_target,
            *args, **kwargs
        )
    if path_source.lower().endswith('.czi'):
        print('Assuming {} is timelapse CZI.'.format(path_source))
        return create_dataset_from_timelapse_czi(
            path_czi=path_source,
            name_signal=name_signal,
            name_target=name_target,
            *args, **kwargs
        )
    raise NotImplementedError

def create_train_test_dataframe_from_timelapse_czi(
        path_czi,
        name_signal,
        name_target,
        train_split=15,
):
    """Create train set and test set dataframes from timelapse CZI with each time slice as a dataset element."""
    assert (name_signal in CHANNEL_TYPES) and (name_target in CHANNEL_TYPES)
    print('reading:', path_czi)
    czi = CziReader(path_czi)
    assert 'T' in czi.axes, 'CZI must have time dimension'
    meta = czi.metadata
    
    # check CZI dimensions
    dims_min = (32, 64, 64)
    shape = _get_shape_from_metadata(meta)
    assert all((shape[i] >= dims_min[i]) for i in range(3))
            
    tag_list = 'Metadata.DisplaySetting.Channels.Channel.attrib'.split('.')
    channel_info_dicts = get_czi_metadata(meta, tag_list)
    chan_idx_signal = _get_chan_idx(name_signal, channel_info_dicts)
    chan_idx_target = _get_chan_idx(name_target, channel_info_dicts)
    
    n_time_slices = czi.get_size('T')
    paths_czi = [path_czi]*n_time_slices
    chans_signal = [chan_idx_signal]*n_time_slices
    chans_target = [chan_idx_target]*n_time_slices
    time_slices = range(n_time_slices)
    
    df_all = pd.DataFrame(
        {
            'path_czi':paths_czi,
            'channel_signal':chans_signal,
            'channel_target':chans_target,
            'time_slice':time_slices,
        })
    df_train, df_test = _shuffle_split_df(df_all, train_split, no_shuffle=True)
    return df_train, df_test

def create_dataset_from_timelapse_czi(
        path_czi,
        name_signal,
        name_target,
        train_split=15,
        transforms=None,
):
    """Create dataset from timelapse CZI with each time slice as a dataset element."""
    czi = CziReader(path_czi)
    assert 'T' in czi.axes, 'CZI must have time dimension'
    meta = czi.metadata
    
    # check CZI dimensions
    dims_min = (32, 64, 64)
    shape = _get_shape_from_metadata(meta)
    assert all((shape[i] >= dims_min[i]) for i in range(3))
            
    tag_list = 'Metadata.DisplaySetting.Channels.Channel.attrib'.split('.')
    channel_info_dicts = get_czi_metadata(meta, tag_list)
    chan_idx_signal = _get_chan_idx(name_signal, channel_info_dicts)
    chan_idx_target = _get_chan_idx(name_target, channel_info_dicts)
    
    n_time_slices = czi.get_size('T')
    paths_czi = [path_czi]*n_time_slices
    chans_signal = [chan_idx_signal]*n_time_slices
    chans_target = [chan_idx_target]*n_time_slices
    time_slices = range(n_time_slices)
    
    df_all = pd.DataFrame(
        {
            'path_czi':paths_czi,
            'channel_signal':chans_signal,
            'channel_target':chans_target,
            'time_slice':time_slices,
        })
    df_train, df_test = _shuffle_split_df(df_all, train_split, no_shuffle=True)
    ds = DataSet(
        df_train,
        df_test,
        transforms,
    )
    return ds

def create_dataset_from_dir(
        path_dir,
        name_signal,
        name_target,
        train_split=15,
        transforms=None,
):
    """Create dataset from directory of CZI files.

    train_split - (int, float) if int, number of images in training set. If float, percent of images in training_set.
    """
    paths_pre = [i.path for i in os.scandir(path_dir) if i.is_file() and i.path.lower().endswith('.czi')]  # order is arbitrary
    print(len(paths_pre), 'files found')
    paths, chans_target, chans_signal = [], [], []
    for i, path in enumerate(paths_pre):
        print('reading:', path)
        czi = CziReader(path)
        meta = czi.metadata
        
        # check CZI dimensions
        dims_min = (32, 64, 64)
        shape = _get_shape_from_metadata(meta)
        if any((shape[i] < dims_min[i]) for i in range(3)):
            print('CZIs dims {} below minimum {}. Skipping: {}'.format(shape, dims_min, path))
            continue

        # find signal/target channel numbers from CZI metadata
        tag_list = 'Metadata.DisplaySetting.Channels.Channel.attrib'.split('.')
        channel_info_dicts = get_czi_metadata(meta, tag_list)
        chan_idx_signal = _get_chan_idx(name_signal, channel_info_dicts)
        chan_idx_target = _get_chan_idx(name_target, channel_info_dicts)
        if (chan_idx_signal is not None) and (chan_idx_target is not None):
            paths.append(path)
            chans_signal.append(chan_idx_signal)
            chans_target.append(chan_idx_target)
        else:
            print('Cannot find channels from metadata. Skipping: {}'.format(path))
            continue
            
    assert len(paths) > 0
    assert len(paths) == len(chans_signal) == len(chans_target)
    df_all = pd.DataFrame(
        {
            'path_czi':paths,
            'channel_signal':chans_signal,
            'channel_target':chans_target,
        })
    df_train, df_test = _shuffle_split_df(df_all, train_split)
    ds = DataSet(
        df_train,
        df_test,
        transforms,
    )
    return ds

def save_dataset(path_save, dataset):
    assert isinstance(dataset, DataSet)
    dirname = os.path.dirname(path_save)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    package = dataset
    with open(path_save, 'wb') as fo:
        pickle.dump(package, fo)
        print('saved dataset to:', path_save)

def load_dataset(path_load):
    with open(path_load, 'rb') as fin:
        package = pickle.load(fin)
    if isinstance(package, DataSet):
        dataset = package
        print('loaded dataset from:', path_load)
    elif isinstance(package, dict):
        if '_df_test' in package:
            dataset = DataSet2(path_load=path_load)
            assert isinstance(dataset, DataSet2)
        else:
            dataset = DataSet1(path_load=path_load)
            assert isinstance(dataset, DataSet1)
    else:
        raise NotImplementedError
    return dataset
