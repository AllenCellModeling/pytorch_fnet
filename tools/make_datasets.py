import argparse
import numpy as np
import os
import pandas as pd
import pdb

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
    #    'dic-lamin',
    #    'dic-membrane',
)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struct', choices=STRUCT_CHOICES, help='target structure')
    parser.add_argument('--train_split', type=int, default=50, help='number of images in training set')
    opts = parser.parse_args()

    name_target = opts.struct
    train_split = opts.train_split

    tag = name_target.lower().replace(' ', '_')
    path_master_csv = '../data/data_jobs_out.csv'

    df = pd.read_csv(path_master_csv).drop_duplicates('inputFilename')
    assert 'path_czi' not in df.columns
    col_path_czi = df['inputFolder'] + os.path.sep + df['inputFilename']
    df = df.assign(path_czi=col_path_czi.values)

    df_target_all = filter_df(df, opts.struct, train_split)
    if opts.struct == 'dna':
        col_name_target = 'nucChannel'
    elif opts.struct == 'membrane':
        col_name_target = 'memChannel'
    else:
        df_target_all = filter_df(df, train_split)
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
    
    df_target_train, df_target_test = shuffle_split_df(
        df_target_all,
        train_split,
    )
    print('train set size:', df_target_train.shape[0], '| test set size:', df_target_test.shape[0])

    # save CSVs
    path_all_csv = os.path.join('../data',  tag + '_all.csv')
    path_train_csv = os.path.join('../data',  tag + '_train.csv')
    path_test_csv = os.path.join('../data',  tag + '_test.csv')
    df_target_all.to_csv(path_all_csv, index=False)
    print('wrote:', path_all_csv)
    df_target_train.to_csv(path_train_csv, index=False)
    print('wrote:', path_train_csv)
    df_target_test.to_csv(path_test_csv, index=False)
    print('wrote:', path_test_csv)
