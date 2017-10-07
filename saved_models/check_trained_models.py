import os
import pandas as pd
import pdb

path_rejects_csv = '../data/dataset_eval/czi_rejects.csv'

def is_model_dir(path):
    for i in os.listdir(path):
        if i.lower().endswith('model.p'):
            return True
    return False

def get_rejects_in_df(df, df_rejects):
    # look for rejected CZIs in df
    s_paths = df['path_czi']
    s_rejects = df_rejects['path_czi']
    mask = s_rejects.isin(s_paths)
    return df_rejects[mask]

def get_df_train(path):
    # path : path to model dir
    for i in os.scandir(path):
        if i.path.lower().endswith('_train.csv'):
            return pd.read_csv(i.path)
    return None
    
def main():
    paths_dirs = [i.path for i in os.scandir('.') if i.is_dir()]
    paths_dirs.sort()

    df_rejects = pd.read_csv(path_rejects_csv)
    for path in paths_dirs:
        if not is_model_dir(path):
            continue
        df_train = get_df_train(path)
        if df_train is None:
            continue
        df_rejects_in_train_set = get_rejects_in_df(df_train, df_rejects)
        pass_fail = 'pass'
        msg = ''
        reasons = None
        if df_rejects_in_train_set.shape[0] > 0:
            pass_fail = 'FAIL'
            msg = '{:d} rejected elements in training set'.format(df_rejects_in_train_set.shape[0])
            reasons = df_rejects_in_train_set['reason'].values
        print('{:s} {:30s} {:s}'.format(pass_fail, os.path.basename(path), msg))
        if reasons is not None:
            for r in reasons:
                print('  ', r)

if __name__ == '__main__':
    main()
