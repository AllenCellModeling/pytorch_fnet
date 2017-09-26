import pandas as pd
import numpy as np
import os

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


if __name__ == '__main__':
    name_target = 'Lamin B1'
    train_split = 30

    tag = name_target.lower().replace(' ', '_')
    path_master_csv = '../data/data_jobs_out.csv'
    path_all_csv = os.path.join('../data',  tag + '_all.csv')
    path_train_csv = os.path.join('../data',  tag + '_train.csv')
    path_test_csv = os.path.join('../data',  tag + '_test.csv')

    df = pd.read_csv(path_master_csv).drop_duplicates('inputFilename')

    mask = df['structureProteinName'] == name_target
    mask &= df['inputFolder'].str.contains('microscopy')

    df_target_all = df[mask]
    col_path_czi = df_target_all['inputFolder'] + os.path.sep + df_target_all['inputFilename']
    df_target_all = df_target_all.assign(path_czi=col_path_czi.values)
    df_target_all = df_target_all.loc[:, ('path_czi', 'lightChannel', 'structureChannel')]
    

    dict_rename = dict(
        lightChannel = 'channel_signal',
        structureChannel = 'channel_target',
    )
    df_target_all = df_target_all.rename(columns=dict_rename)
    # change channels to be zero-indexed
    df_target_all['channel_signal'] = df_target_all['channel_signal'] - 1
    df_target_all['channel_target'] = df_target_all['channel_target'] - 1
    
    df_target_train, df_target_test = shuffle_split_df(
        df_target_all,
        train_split,
    )

    df_target_all.to_csv(path_all_csv, index=False)
    print('wrote:', path_all_csv)
    df_target_train.to_csv(path_train_csv, index=False)
    print('wrote:', path_train_csv)
    df_target_test.to_csv(path_test_csv, index=False)
    print('wrote:', path_test_csv)
