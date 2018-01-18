import os
import pandas as pd
import tifffile
import numpy as np
import fnet.transforms
from fnet.data.czidataset import CziDataset
from fnet.data.dummychunkdataset import DummyChunkDataset
import torch.utils.data
import sys
import pdb

def load_dataset(
        path_csv: str,
        class_dataset: torch.utils.data.Dataset,
        path_store_split: str,
        use_local: bool = True,
        train: bool = True,
        train_size: float = 0.8,
        shuffle: bool = True,
        random_seed: int = 0,
        **kwargs
):
    path_train_csv = os.path.join(path_store_split, 'train.csv')
    path_test_csv = os.path.join(path_store_split, 'test.csv')
    if (not use_local or
        not os.path.exists(path_train_csv) and
        not os.path.exists(path_test_csv)
    ):
        rng = np.random.RandomState(random_seed)
        df_all = pd.read_csv(path_csv)
        if shuffle:
            df_all = df_all.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        if train_size == 0:
            df_test = df_all
            df_train = df_all[0:0]  # empty DataFrame but with columns intact
        else:
            if isinstance(train_size, int):
                idx_split = train_split
            elif isinstance(train_size, float):
                idx_split = round(len(df_all)*train_size)
            else:
                raise AttributeError
        df_train = df_all[:idx_split]
        df_test = df_all[idx_split:]
        if not os.path.exists(path_store_split):
            os.makedirs(path_store_split)
        df_train.to_csv(path_train_csv, index=False)
        df_test.to_csv(path_test_csv, index=False)
    else:
        df_train = pd.read_csv(path_train_csv)
        df_test = pd.read_csv(path_test_csv)
    print('DEBUG: train/test', len(df_train), len(df_test))
    df = df_train if train else df_test
    ds = class_dataset(
        df,
        **kwargs,
    )
    return ds

def load_alpha_tubulin(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_beta_actin(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_desmoplakin(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_dna(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
# def load_fibrillarin(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_lamin_b1(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_membrane(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
# def load_myosin_iib(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_sec61_beta(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_st6gal1(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_tom20(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)
def load_zo1(**kwargs): return _load_structure(sys._getframe().f_code.co_name.split('load_')[-1], **kwargs)

def load_dummy_chunk(**kwargs):
    ds = DummyChunkDataset(
        dims_chunk = (1, 64, 128, 128),
    )
    return ds

def _load_structure(structure, **kwargs):
    """For CZI-based, microscopy pipeline datasets."""
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    resizer = fnet.transforms.Resizer((1, factor_yx, factor_yx))
    kwargs_czidataset = dict(
        transform_source = [fnet.transforms.normalize, resizer],
        transform_target = [fnet.transforms.normalize, resizer],
    )
    kwargs.update(kwargs_czidataset)
    return load_dataset(
        path_csv = 'data/csvs/{:s}_czis.csv'.format(structure),
        class_dataset = CziDataset,
        path_store_split = 'data/{:s}'.format(structure),
        **kwargs,
    )

def _test():
    rng = np.random.RandomState(666)
    path_out_dir = 'outputs'
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)
    ds = load_dna()
    print('len:', len(ds))
    for idx in rng.randint(0, len(ds), size=5):
        for i_part, part in enumerate(ds[idx]):
            print('index: {} | shape: {}'.format(idx, part.shape))
            # path_tiff = os.path.join(path_out_dir, '{:04d}_{:d}.tiff'.format(idx, i_part))
            # tifffile.imsave(path_tiff, part)

if __name__ == '__main__':
    _test()
