import pickle
import os
from util.data.czireader import CziReader
from util.data.dataset3 import DataSet3
import pdb
import pandas as pd

_channel_types = ('bf', 'dic', 'dna', 'memb', 'struct')

def get_metadata(element, tag_list):
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
                retval = get_metadata(sub_ele, tag_list[1:])
                if retval is not None:
                    values.extend(retval)
    if len(values) == 0:
        return None
    return values

def _shuffle_split_df(df_all, train_split):
    df_all_shuf = df_all.sample(frac=1).reset_index(drop=True)
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

def create_dataset_from_dir(
        path_dir,
        name_signal,
        name_target,
        train_split=15,
        transforms=None,
):
    """Create dataset from CZI files in directory


    train_split - (int, float) if int, number of images in training set. If float, percent of images in training_set.
    """
    # ('bf', 'dic', 'dna', 'memb', 'struct')
    def get_chan_idx(channel_nickname, channel_info_dicts):
        retval = None
        for info_dict in channel_info_dicts:
            name = info_dict.get('Name')
            name_match = False
            if channel_nickname == 'bf':
                raise NotImplementedError
            elif channel_nickname == 'dic':
                if name.endswith('DIC'):
                    name_match = True
            elif channel_nickname == 'dna':
                raise NotImplementedError
            elif channel_nickname == 'memb':
                if name == 'CMDRP':
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
    
    assert name_signal in _channel_types
    assert name_target in _channel_types
    
    paths_pre = [i.path for i in os.scandir(path_dir) if i.is_file() and i.path.lower().endswith('.czi')]  # order is arbitrary
    print(len(paths_pre), 'files found')
    paths, chans_target, chans_signal = [], [], []
    for i, path in enumerate(paths_pre):
        czi = CziReader(path)
        meta = czi.metadata
        tag_list = 'Metadata.DisplaySetting.Channels.Channel.attrib'.split('.')
        channel_info_dicts = get_metadata(meta, tag_list)
        chan_idx_signal = get_chan_idx(name_signal, channel_info_dicts)
        chan_idx_target = get_chan_idx(name_target, channel_info_dicts)
        if (chan_idx_signal is not None) and (chan_idx_target is not None):
            paths.append(path)
            chans_signal.append(chan_idx_signal)
            chans_target.append(chan_idx_target)
    assert len(paths) > 0
    assert len(paths) == len(chans_signal) == len(chans_target)
    df_all = pd.DataFrame(
        {
            'path_czi':paths,
            'channel_signal':chan_idx_signal,
            'channel_target':chan_idx_target,
        })
    df_train, df_test = _shuffle_split_df(df_all, train_split)
    df = DataSet3(
        df_train,
        df_test,
        transforms,
    )
    return df

def save_dataset(path_save, dataset):
    assert isinstance(dataset, DataSet3)
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
    print('loaded dataset from:', path_load)
    assert isinstance(package, DataSet3)
    dataset = package
    return dataset
