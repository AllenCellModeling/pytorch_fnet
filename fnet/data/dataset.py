import os
import pickle
import numpy as np
from fnet import get_vol_transformed
import pandas as pd
import pdb
import collections
import warnings
from fnet.data.czireader import CziReader
from fnet.transforms import Resizer

class DataSet(object):
    def __init__(self,
                 path_train_csv,
                 path_test_csv,
                 scale_z = 0.3,
                 scale_xy = 0.3,
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
        self._czi = None
        self._last_loaded = None

    def use_train_set(self):
        self._train_select = True
        self._df_active = self.df_train
        
    def use_test_set(self):
        self._train_select = False
        self._df_active = self.df_test

    def is_timelapse(self):
        return 'time_slice' in self._df_active.columns

    def __len__(self):
        return len(self._df_active)

    def get_name(self, idx, *args):
        return self._df_active['path_czi'].iloc[idx]

    def get_item_sel(self, idx, sel, apply_transforms=True):
        """Get item(s) from dataset element idx.

        DataFrames should have columns ('path_czi', 'channel_signal', 'channel_target') and optionally 'time_slice'.

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
        if ('_last_loaded' not in vars(self)) or self._last_loaded != path:

            print('reading:', path)
            try:
                self._czi = CziReader(path)
                self._last_loaded = path
            except Exception as e:
                warnings.warn('could not read file: {}'.format(path))
                warnings.warn(str(e))
                return None
        
        time_slice = None
        if 'time_slice' in self._df_active.columns:
            time_slice = self._df_active['time_slice'].iloc[idx]
        dict_scales = self._czi.get_scales()
        scales_orig = [dict_scales.get(dim) for dim in 'zyx']
        # print('pixel scales:', scales_orig)
        
        if self.scale_z is not None or self.scale_xy is not None:
            if None in scales_orig:
                warnings.warn('bad pixel scales in {:s} | scales: {:s}'.format(path, str(scales_orig)))
                return None
            scales_wanted = [self.scale_z, self.scale_xy, self.scale_xy]
            factors_resize = list(map(lambda a, b : a/b if None not in (a, b) else 1.0, scales_orig, scales_wanted))
            # print('factors_resize:', factors_resize)
            resizer = Resizer(factors_resize)
        else:
            resizer = None
        
        volumes = []
        for i in range(len(sels)):
            if sels[i] == 0:
                chan = self._df_active['channel_signal'].iloc[idx]
            else:
                chan = self._df_active['channel_target'].iloc[idx]
            volume_pre = self._czi.get_volume(chan, time_slice=time_slice)
            if not apply_transforms:
                volumes.append(volume_pre)
            else:
                transforms = []
                if resizer is not None:
                    transforms.append(resizer)
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
        """Returns arrays corresponding to files identified by file_tags in the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - n-element tuple or None. If the file read was successful, return tuple
                  of transformed arrays else return None
        """
        return self.get_item_sel(idx, (0, 1), apply_transforms=True)

    
if __name__ == '__main__':
    raise NotImplementedError
