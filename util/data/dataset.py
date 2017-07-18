import os
import pickle
import glob
from aicsimage.io import omeTifReader
import numpy as np
from natsort import natsorted
from util import get_vol_transformed

class DataSet(object):
    def __init__(self, path, train, percent_test=0.1, train_set_limit=None, transform=None, target_transform=None, force_rebuild=False):
        """
        path - path to parent directory that contains folders of data.
        train - (bool) if True, creates a dataset from 
        percent_test - (float between 0.0 and 1.0) percent of folders to be used as test set
        """
        self._path = path
        self._train_select = train
        self._percent_test = percent_test
        self._train_set_limit = train_set_limit
        self._transform = transform
        self._target_transform = target_transform
        self._all_set = None
        self._test_set = None
        self._train_set = None
        
        self._save_path = os.path.join(os.path.dirname(path), 'dataset_saves', os.path.basename(path) + '.p')
        if not os.path.exists(self._save_path) or force_rebuild:
            self._build_new_sets()
            self._save()
        else:
            self._load()
        if self._train_select:
            self._active_set = self._train_set
        else:
            self._active_set = self._test_set
        self._active_set = natsorted(self._active_set)  # TODO: is this wanted?
        self._validate_dataset()

    def _build_new_sets(self):
        self._all_set = [i.path for i in os.scandir(self._path) if i.is_dir()]  # order is arbitrary
        # TODO: shuffle self._all_set
        if len(self._all_set) == 1:
            print('WARNING: DataSet has only one element')
            self._test_set = self._all_set
            self._train_set = self._all_set
        else:
            if self._train_set_limit is not None:
                idx_split = self._train_set_limit*-1
                print('limiting training set to {:d} elements'.format(self._train_set_limit))
            else:
                idx_split = round(len(self._all_set)*self._percent_test)
            self._test_set = self._all_set[:idx_split]
            self._train_set = self._all_set[idx_split:]
    
    def get_active_set(self):
        return self._active_set
    
    def get_test_set(self):
        return self._test_set
    
    def get_train_set(self):
        return self._train_set

    def _validate_dataset(self):
        assert self._active_set is not None
        # TODO: check folders?

    def _save(self):
        dirname = os.path.dirname(self._save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        package = (self._path, self._percent_test, self._train_set_limit, self._transform, self._target_transform,
                   self._all_set, self._test_set, self._train_set)
        assert len(package) == 8
        with open(self._save_path, 'wb') as fo:
            pickle.dump(package, fo)
            print('saved dataset split to:', self._save_path)

    def _load(self):
        with open(self._save_path, 'rb') as fin:
            package = pickle.load(fin)
        if len(package) == 7:
            # TODO: added to make compatible with older datasets
            (self._path, self._percent_test, self._train_set_limit, self._transform,
             self._all_set, self._test_set, self._train_set) = package
            self._target_transform = self._transform
        if len(package) == 8:
            (self._path, self._percent_test, self._train_set_limit, self._transform, self._target_transform,
             self._all_set, self._test_set, self._train_set) = package
        print('loaded dataset split from:', self._save_path)

    def __str__(self):
        def get_str_transform(transform):
            # Return the string representation of the given transform
            if transform is None:
                str_transform = str(None)
            elif isinstance(transform, (list, tuple)):
                str_list = []
                for t in transform:
                    str_list.append(str(t))
                str_transform = ' => '.join(str_list)
            else:
                str_transform = str(transform)
            return str_transform
            
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('DataSet from: ' + self._path)
        str_list.append('active_set: ' + str_active)
        str_list.append('percent_test: ' + str(self._percent_test))
        str_list.append('train_set_limit: ' + str(self._train_set_limit))
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._train_set),
                                                                  len(self._test_set),
                                                                  len(self._all_set)))
        str_list.append('transform: ' + get_str_transform(self._transform))
        str_list.append('target_transform: ' + get_str_transform(self._target_transform))
        return os.linesep.join(str_list)


    def __len__(self):
        return len(self._active_set)

    def __getitem__(self, index):
        """Returns arrays corresponding to the transmitted light and DNA channels of the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - 2-element tuple or None. If the file read was successful, return tuple
        (array of the trans channel, array of the DNA channel) else return None
        """
        path_folder = self._active_set[index]
        volumes_pre = _read_tifs(path_folder)  # TODO: add option to read different file types
        if volumes_pre is None:
            return None
        volumes = (get_vol_transformed(volumes_pre[0], self._transform),
                   get_vol_transformed(volumes_pre[1], self._target_transform))
        # print('DEBUG: DataSet. item shapes:', volumes[0].shape, volumes[1].shape)
        return volumes
    
if __name__ == '__main__':
    pass
