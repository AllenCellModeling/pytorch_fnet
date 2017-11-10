import sys
sys.path.append('.')
import argparse
import numpy as np
import os
import pandas as pd
import pdb
import fnet.data.czireader as czireader
import re

STRUCT_CHOICES = (
    'Alpha tubulin',
    'Beta actin',
    'Desmoplakin',
    'Fibrillarin',
    'Lamin B1',
    'Myosin IIB',
    'Sec61 beta',
    'Tom20',
    'ZO1',
    'dna',
    'membrane',
    'dic-lamin_b1',
    'dic-membrane',
)

def get_channel_mapping_from_czi_metadata(metadata):
    tag_list = 'Metadata.Information.Image.Dimensions.Channels.Channel.attrib'.split('.')
    attribs_channels = czireader.get_czi_metadata(metadata, tag_list)
    # attribs_channels = [{'Id': 'Channel:0', 'Name': 'EGFP'}, {'Id': 'Channel:1', 'Name': 'TL Brightfield'}]  # tester
    pattern = re.compile(r'channel:(\d+)', re.IGNORECASE)
    mapping_channels = {}
    for attrib in attribs_channels:
        id_entry = attrib.get('Id')
        name_entry = attrib.get('Name')
        if id_entry is not None:
            match = pattern.search(id_entry)
            if match is not None:
                mapping_channels[name_entry] = int(match.groups()[0])
    return mapping_channels

def get_dic_lamin_b1_dataset():
    path_dir = '/allen/aics/microscopy/PRODUCTION/PIPELINE_4_OptimizationAutomation/DICLAMINB/ZSD2/100X_zstack'
    paths_czis = [i.path for i in os.scandir(path_dir) if i.path.lower().endswith('.czi')]
    paths_czis.sort()
    s_paths_czis = pd.Series(paths_czis)
    mask = ~s_paths_czis.str.startswith('//')
    s_paths_czis[mask] = '/' + s_paths_czis[mask]  # add starting double slash
    df_all = pd.DataFrame({
        'path_czi': s_paths_czis,
        'channel_signal': 6,
        'channel_target': 3,
    })
    return df_all

def get_dic_membrane_dataset():
    path_dir = '/allen/aics/microscopy/PRODUCTION/PIPELINE_4_OptimizationAutomation/DICLAMINB/ZSD2/100X_zstack'
    paths_czis = [i.path for i in os.scandir(path_dir) if i.path.lower().endswith('.czi')]
    paths_czis.sort()
    s_paths_czis = pd.Series(paths_czis)
    mask = ~s_paths_czis.str.startswith('//')
    s_paths_czis[mask] = '/' + s_paths_czis[mask]  # add starting double slash
    df_all = pd.DataFrame({
        'path_czi': s_paths_czis,
        'channel_signal': 6,
        'channel_target': 1,
    })
    return df_all

def shuffle_split_df(df_all, train_split, no_shuffle=False):
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

def filter_df(df, target, min_elements):
    # define masks in order of priority
    path_csv_rejects = './czi_rejects.csv'
    paths_rejects = []
    if os.path.exists(path_csv_rejects):
        df_rejects = pd.read_csv(path_csv_rejects)
        paths_rejects = df_rejects['path_czi']
    masks = []
    masks.append( ('no rejects', ~df['path_czi'].isin(paths_rejects) ) )
    if target == 'dna':
        masks.append( ('microscopy only', df['inputFolder'].str.lower().str.contains('aics/microscopy')) )
        masks.append( ('no minipipeline', ~df['inputFolder'].str.lower().str.contains('minipipeline')) )
    elif target == 'membrane':
        masks.append( ('microscopy only', df['inputFolder'].str.lower().str.contains('aics/microscopy')) )
        masks.append( ('no minipipeline', ~df['inputFolder'].str.lower().str.contains('minipipeline')) )
    else:
        masks.append( ('microscopy only', df['inputFolder'].str.lower().str.contains('aics/microscopy')) )
        masks.append( ('no minipipeline', ~df['inputFolder'].str.lower().str.contains('minipipeline')) )
    
    # apply as many masks as possible to get desired training set size
    for n_masks in range(len(masks), 0, -1):
        mask_combi = masks[0][1]
        names_masks = [masks[0][0]]
        for i in range(1, n_masks):
            mask_combi &= masks[i][1]
            names_masks.append(masks[i][0])
        df_target_all = df[mask_combi]
        print('found {} images using mask {}'.format(df_target_all.shape[0], str(names_masks)))
        if df_target_all.shape[0] > min_elements:
            break
    return df_target_all

def get_df_from_csv_archive():
    pass
    tag = name_target.lower().replace(' ', '_')
    path_master_csv = os.path.join(
        os.path.dirname(os.path.realpath(sys.argv[0])),
        'data/data_jobs_out.csv',
    )

    df = pd.read_csv(path_master_csv).drop_duplicates('inputFilename')
    assert 'path_czi' not in df.columns
    col_path_czi = df['inputFolder'] + os.path.sep + df['inputFilename']
    df = df.assign(path_czi=col_path_czi.values)

    df_target_all = filter_df(df, opts.struct, n_elements_min)

    if opts.struct == 'dna':
        col_name_target = 'nucChannel'
    elif opts.struct == 'membrane':
        col_name_target = 'memChannel'
    else:
        col_name_target = 'structureChannel'

    df_target_all = df_target_all.loc[:, ('path_czi', 'lightChannel', col_name_target)]
    dict_rename = {
        'lightChannel': 'channel_signal',
        col_name_target: 'channel_target',
    }
    df_target_all = df_target_all.rename(columns=dict_rename)
    # change channels to be zero-indexed
    df_target_all['channel_signal'] = df_target_all['channel_signal'] - 1
    df_target_all['channel_target'] = df_target_all['channel_target'] - 1

def get_df_from_czi(path_czi, tag_signal, tag_target):
    """
    tag_signal - string to identify signal channel
    tag_target - string to identify target channel
    """
    czi = czireader.CziReader(path_czi)
    size_time = None
    if 'T' in czi.axes:
        size_time = czi.get_size('T')
    mapping_channels = get_channel_mapping_from_czi_metadata(czi.metadata)
    # mapping_channels = {'TL Brightfield': 1, 'EGFP': 0, 'EGFP_1': 3}  # tester
    pattern_dummy = re.compile(r'_\d+$')
    channel_signal = None
    channel_target = None
    for name_channel, channel in mapping_channels.items():
        if pattern_dummy.search(name_channel) is not None:  # ignore dummy channels
            continue
        if tag_signal in name_channel.lower():
            channel_signal = channel
        if tag_target in name_channel.lower():
            channel_target = channel
    assert channel_signal is not None and channel_target is not None
    cols_df = {}
    cols_df['path_czi'] = path_czi
    cols_df['channel_signal'] = channel_signal
    cols_df['channel_target'] = channel_target
    if size_time is not None and size_time > 1:
        cols_df['time_slice'] = range(size_time)
    df = pd.DataFrame(cols_df)
    return df

def name_unified(s):
    return s.lower().replace(' ', '_')

def get_df_from_csv(path_csv, line, tag_signal, tag_target):
    """
    line - tag to identify cell line
    tag_signal - string to identify signal channel
    tag_target - string to identify target channel
    """
    df_source = pd.read_csv(path_csv)
    mask = df_source['structureProteinName'].str.lower().str.replace(' ', '_') == name_unified(line)
    df_line = df_source[mask]
    cols_take = []
    cols_take.append('path_czi')
    cols_take.append('structureProteinName')
    if name_unified(tag_signal) == 'brightfield':
        cols_take.append('lightChannel')
    if name_unified(tag_target) == 'egfp':
        col_name_target = 'structureChannel'
        cols_take.append('structureChannel')
    df = df_line.loc[:, cols_take]
    dict_rename = {
        'structureProteinName': 'cell_line',
        'lightChannel': 'channel_signal',
        col_name_target: 'channel_target',
    }
    df = df.rename(columns=dict_rename)
    # adjust channel numbers for zero-indexing
    df['channel_signal'] = df['channel_signal'] - 1
    df['channel_target'] = df['channel_target'] - 1
    return df
    
def get_df_from_dir(path_dir, tag_signal, tag_target):
    """
    path_dir
    tag_signal - string to identify signal channel
    tag_target - string to identify target channel
    """
    assert os.path.isdir(path_dir)
    paths_czis = sorted([i.path for i in os.scandir(path_dir) if i.path.lower().endswith('.czi')])
    print(paths_czis)
    path = paths_czis[0]
    df = get_df_from_czi(path, tag_signal, tag_target)
    covfefe
    
def get_ds_name(signal, target, path_source):
    """Generate a name for dataset depending on target and source file/directory."""
    name_signal = name_unified(signal)
    name_target = name_unified(target)
    tail = path_source.split('.')[-1].lower()
    if tail in ['czi', 'csv']:
        name_source = os.path.basename(path_source)
    name = '{:s}_to_{:s}_from_{:s}'.format(name_signal, name_target, name_source)
    return name

def main():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal', default='bright', help='input image type')
    parser.add_argument('--target', default='dna', help='target structure')
    parser.add_argument('--path_source', help='CZI, directory of CZIs, or CSV file')
    parser.add_argument('--path_save_dir', default='data', help='directory to save dataset')
    opts = parser.parse_args()

    print('creating dataset from:', opts.path_source)
    assert os.path.exists(opts.path_source)

    tag_signal = opts.signal
    if opts.target == 'dna':
        raise NotImplementedError
    elif opts.target == 'membrane':
        tag_target = 'membrane'
    else:
        tag_target = 'egfp'
    if opts.path_source.lower().endswith('.czi'):
        df = get_df_from_czi(opts.path_source, tag_signal, tag_target)
    elif opts.path_source.lower().endswith('.csv'):
        line = tag_signal
        df = get_df_from_csv(opts.path_source, line, tag_signal, tag_target)
    elif os.path.isdir(opts.path_source):
        df = get_df_from_dir(opts.path_source, opts.signal, opts.target)
    else:
        raise NotImplementedError
    
    # save CSVs
    if not os.path.exists(opts.path_save_dir):
        os.makedirs(opts.path_save_dir)
    path_ds = os.path.join(
        opts.path_save_dir,
        get_ds_name(opts.signal, opts.target, opts.path_source) + '.csv',
    )
    df.to_csv(path_ds, index=False)
    print('wrote:', path_ds)

if __name__ == '__main__':
    main()
