import os
import pickle
import numpy as np
from util import get_vol_transformed
import pandas as pd
import pdb
import collections
import warnings
from util.data.czireader import CziReader

class DataSet3(object):
    def __init__(self,
                 df_train,
                 df_test,
                 transforms=None):
        """Create dataset from train/test DataFrames.
        
        df_train - pandas.DataFrame with columns ('path_czi', 'channel_signal', 'channel_target')
        df_test - pandas.DataFrame, same columns as above
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        self._df_train = df_train
        self._df_test = df_test
        self._transforms = transforms
        self._train_select = True
        self._df_active = self._df_train

    def use_train_set(self):
        self._train_select = True
        self._df_active = self._df_train
        
    def use_test_set(self):
        self._train_select = False
        self._df_active = self._df_test

    def __len__(self):
        return len(self._df_active)

    def get_name(self, idx, *args):
        return os.path.basename(self._df_active['path_czi'].iloc[idx])

    def get_item_sel(self, idx, sel, apply_transforms=True):
        """Get item(s) from dataset element idx.

        idx - (int) dataset element index
        sel - (int or iterable) 0 for 'signal', 1 for 'target'
        """
        if isinstance(sel, int):
            assert sel >= 0
            sels = (sel, )
        elif isinstance(sel, collections.Iterable):
            sels = sel
        else:
            raise AttributeError
        path = self._df_active['path_czi'].iloc[idx]
        try:
            print('reading:', path)
            czi = CziReader(path)
        except:
            warnings.warn('could not read file: {}'.format(path))
            return None
        volumes = []
        for i in range(len(sels)):
            if sels[i] == 0:
                chan = self._df_active['channel_signal'].iloc[idx]
            else:
                chan = self._df_active['channel_target'].iloc[idx]
            volume_pre = czi.get_volume(chan)
            if self._transforms is None or not apply_transforms:
                volumes.append(volume_pre)
            else:
                volumes.append(get_vol_transformed(volume_pre, self._transforms[sels[i]]))
        return volumes[0] if isinstance(sel, int) else volumes

    def __str__(self):
        def get_str_transform(transforms):
            # Return the string representation of the given transforms
            if transforms is None:
                return str(None)
            all_transforms = []
            for transform in transforms:
                if transform is None:
                    all_transforms.append(str(None))
                elif isinstance(transform, (list, tuple)):
                    str_list = []
                    for t in transform:
                        str_list.append(str(t))
                    all_transforms.append(' => '.join(str_list))
                else:
                    all_transforms.append(str(transform))
            return (os.linesep + '            ').join(all_transforms)
        if id(self._df_train) == id(self._df_test):
            n_unique = self._df_train.shape[0]
        else:
            n_unique = self._df_train.shape[0] + self._df_test.shape[0]
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('{}:'.format(self.__class__.__name__))
        str_list.append('active_set: ' + str_active)
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._df_train),
                                                                  len(self._df_test),
                                                                  n_unique))
        str_list.append('transforms: ' + get_str_transform(self._transforms))
        return os.linesep.join(str_list)

    def __getitem__(self, idx):
        """Returns arrays corresponding to files identified by file_tags in the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - n-element tuple or None. If the file read was successful, return tuple
                  of transformed arrays else return None
        """
        return self.get_item_sel(idx, (0, 1), apply_transforms=True)

    
DataSet = DataSet3

if __name__ == '__main__':
    raise NotImplementedError
