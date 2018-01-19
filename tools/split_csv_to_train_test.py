import argparse
import numpy as np
import os
import pandas as pd
import pdb

def shuffle_split_df(df_all, train_split, no_shuffle=False):
    if not no_shuffle:
        df_all_shuf = df_all.sample(frac=1, random_state=0).reset_index(drop=True)
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

def clean_df(df):
    # check for duplicates, use only passing files, remove 'pass',  column
    df_cleaned = df.drop_duplicates(subset='path_czi')
    if df.shape[0] != df_cleaned.shape[0]:
        print('WARNING: duplicates detected. start shape:', df.shape, '| after shape:', df_cleaned.shape)
    df_cleaned = df_cleaned[df_cleaned['pass'] == True]
    # df_cleaned.drop(['pass', 'reason'], axis=1, inplace=True)
    return df_cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_save_dir', help='path to save directory')
    parser.add_argument('--path_source_csv', help='path to source csv with pass column')
    parser.add_argument('--name', help='name of new dataset')
    parser.add_argument('--train_split', type=int, default=33, help='number of images in training set')
    parser.add_argument('--no_shuffle', action='store_true', help='set to not shuffle source csv')
    opts = parser.parse_args()

    path_train_csv = os.path.join(opts.path_save_dir, '{:s}_train.csv'.format(opts.name))
    path_test_csv = os.path.join(opts.path_save_dir, '{:s}_test.csv'.format(opts.name))
    if not os.path.exists(opts.path_save_dir):
        os.makedirs(opts.path_save_dir)
    
    df_source = pd.read_csv(opts.path_source_csv)
    print('read csv:', opts.path_source_csv)
    df_cleaned = clean_df(df_source)
    df_train, df_test = shuffle_split_df(df_cleaned, opts.train_split, no_shuffle=opts.no_shuffle)
    df_train.to_csv(path_train_csv, index=False)
    print('wrote csv:', path_train_csv)
    df_test.to_csv(path_test_csv, index=False)
    print('wrote csv:', path_test_csv)
    print('*** Total files/train/test: {:d}/{:d}/{:d} ***'.format(df_cleaned.shape[0], df_train.shape[0], df_test.shape[0]))

if __name__ == '__main__':
    main()
