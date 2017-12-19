import os
import numpy as np
from fnet import get_vol_transformed
import pandas as pd
import pdb
import collections
import warnings
import tifffile
from fnet.data.transforms import Resizer

class DataSet(object):
    def __init__(self,
                 path_train_csv,
                 path_test_csv,
                 scale_z = None,
                 scale_xy = None,
                 transforms=None
    ):
        """Create dataset from train/test DataFrames.
        
        Parameters:
        df_train - pandas.DataFrame, where each row is a DataSet element
        df_test - pandas.DataFrame, same columns as above
        scale_z - desired um/px size for z-dimension
        scale_xy - desired um/px size for x, y dimensions
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        self.df_train = pd.read_csv(path_train_csv) if path_train_csv is not None else pd.DataFrame()
        self.df_test = pd.read_csv(path_test_csv) if path_test_csv is not None else pd.DataFrame()
        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self.transforms = transforms
        self._train_select = True
        self._df_active = self.df_train

    def use_train_set(self):
        self._train_select = True
        self._df_active = self.df_train
        
    def use_test_set(self):
        self._train_select = False
        self._df_active = self.df_test

    def __len__(self):
        return len(self._df_active)

    def get_name(self, idx, *args):
        return self._df_active['path_signal'].iloc[idx]

    def get_item_sel(self, idx, sel, apply_transforms=True):
        """Get item(s) from dataset element idx.

        DataFrames should have columns 'path_signal', 'path_target'

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
        
        volumes = []
        for i in range(len(sels)):
            if sels[i] == 0:
                path_source = self._df_active['path_signal'].iloc[idx]
            elif sels[i] == 1:
                path_source = self._df_active['path_target'].iloc[idx]
            print('reading:', path_source)
            if not isinstance(path_source, str):
                volumes.append(None)
                continue
            try:
                volume_pre = tifffile.imread(path_source)
            except Exception as e:
                warnings.warn('could not read file: {}'.format(path_source))
                warnings.warn(str(e))
                return None
            if not apply_transforms:
                volumes.append(volume_pre)
            else:
                transforms = []
                if self.transforms is not None:
                    if isinstance(self.transforms[sels[i]], collections.Iterable):
                        transforms.extend(self.transforms[sels[i]])
                    else:
                        transforms.append(self.transforms[sels[i]])
                volumes.append(get_vol_transformed(volume_pre, transforms))
        return volumes[0] if isinstance(sel, int) else volumes

    def __repr__(self):
        return 'DataSet({:d} train elements, {:d} test elements)'.format(len(self.df_train), len(self.df_test))

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
        if id(self.df_train) == id(self.df_test):
            n_unique = self.df_train.shape[0]
        else:
            n_unique = self.df_train.shape[0] + self.df_test.shape[0]
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('{}:'.format(self.__class__.__name__))
        str_list.append('active_set: ' + str_active)
        str_list.append('scale_z: ' + str(self.scale_z) + ' um/px')
        str_list.append('scale_xy: ' + str(self.scale_xy) + ' um/px')
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self.df_train),
                                                                  len(self.df_test),
                                                                  n_unique))
        str_list.append('transforms: ' + get_str_transform(self.transforms))
        return os.linesep.join(str_list)

    def __getitem__(self, idx):
        return self.get_item_sel(idx, (0, 1), apply_transforms=True)

    
if __name__ == '__main__':
    raise NotImplementedError
