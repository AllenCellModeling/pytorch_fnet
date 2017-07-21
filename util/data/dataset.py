import os
import pickle
import glob
from aicsimage.io import omeTifReader
import numpy as np
from natsort import natsorted
from util import get_vol_transformed
import pdb

class DataSet(object):
    def __init__(self, path, file_tags=None,
                 train_select=True, percent_test=None, train_set_size=None,
                 transforms=None):
        """Load or build new data set.

        If either percent_test or train_set_size is set, a new data set will be created and saved.

        path - path to parent directory that contains folders of data.
        file_tags - (list or tuple or strings) string "tags" that can uniquely identify files in the directory specified by path
        train_select - (bool) if True, DataSet returns elements from training set, otherwise returns elements from test set
        percent_test - (float between 0.0 and 1.0) percent of folders to be used as test set
        train_set_limit - (int) number of elements in training set. If set, this will be used instead of percent_test.
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        self._path = path
        self._file_tags = file_tags
        self._train_select = train_select
        self._percent_test = percent_test
        self._train_set_size = train_set_size
        self._transforms = transforms

        self._all_set = None
        self._test_set = None
        self._train_set = None
        
        self._save_path = os.path.join(os.path.dirname(path), 'dataset_saves', os.path.basename(path) + '.p')
        if not os.path.exists(self._save_path) or self._percent_test or self._train_set_size:
            self._build_new_sets()
            self._save()
        else:
            self._load()
            self._train_select = train_select  # use train_select from parameter ctor
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
            if self._train_set_size is not None:
                idx_split = self._train_set_size*-1
                print('setting training set size to to {:d} elements'.format(self._train_set_size))
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
        assert self._file_tags is not None, 'must specify file tags'
        if self._transforms is not None:
            assert len(self._file_tags) == len(self._transforms)
        # TODO: check folders?

    def _save(self):
        dirname = os.path.dirname(self._save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        package = vars(self)
        with open(self._save_path, 'wb') as fo:
            pickle.dump(package, fo)
            print('saved dataset to:', self._save_path)

    def _load(self):
        with open(self._save_path, 'rb') as fin:
            package = pickle.load(fin)
        # TODO: remove if statements eventually
        if isinstance(package, dict):
            self.__dict__.update(package)
        elif len(package) == 7:
            print('WARNING: legacy DataSet. Rebuild recommended.')
            (self._path, self._percent_test, self._train_set_size, self._transform,
             self._all_set, self._test_set, self._train_set) = package
            self._target_transform = self._transform
        elif len(package) == 8:
            print('WARNING: legacy DataSet. Rebuild recommended.')
            (self._path, self._percent_test, self._train_set_size, self._transform, self._target_transform,
             self._all_set, self._test_set, self._train_set) = package
        print('loaded dataset split from:', self._save_path)

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
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('{} from: {}'.format(self.__class__.__name__, self._path))
        str_list.append('file_tags: ' + str(self._file_tags))
        str_list.append('active_set: ' + str_active)
        if self._train_set_size:
            str_list.append('train_set_size: ' + str(self._train_set_size))
        else:
            str_list.append('percent_test: ' + str(self._percent_test))
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._train_set),
                                                                  len(self._test_set),
                                                                  len(self._all_set)))
        str_list.append('transforms: ' + get_str_transform(self._transforms))
        return os.linesep.join(str_list)

    def __len__(self):
        return len(self._active_set)

    def __getitem__(self, index):
        """Returns arrays corresponding to files identified by file_tags in the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - n-element tuple or None. If the file read was successful, return tuple
                  of transformed arrays else return None
        """
        path_folder = self._active_set[index]
        volumes_pre = _read_tifs(path_folder, self._file_tags)
        if volumes_pre is None:
            return None
        volumes = []
        for i, volume_pre in enumerate(volumes_pre):
            volumes.append(get_vol_transformed(volume_pre, self._transforms[i]))
        return tuple(volumes)

def _read_tifs(path_dir, file_tags):
    """Read TIFs in folder and return as tuple of numpy arrays.

    Parameters:
    path_dir - path to directory of TIFs
    file_tags - tuple/list of strings that should match the ends of the target filenames
                       e.g., (_trans.tif, _dna.tif)

    Returns:
    tuple of numpy arrays
    """
    assert isinstance(file_tags, (tuple, list))
    print('reading TIFs from', path_dir)
    file_list = [i.path for i in os.scandir(path_dir) if i.is_file()]  # order is arbitrary

    paths_to_read = []
    for tag in file_tags:
        matches = [f for f in file_list if (tag in f)]
        if len(matches) != 1:
            print('WARNING: incorrectect number of files found for pattern {} in {}'.format(bit, path_dir))
            return None
        paths_to_read.append(matches[0])
    
    vol_list = []
    for path in paths_to_read:
        fin = omeTifReader.OmeTifReader(path)
        try:
            fin = omeTifReader.OmeTifReader(path)
        except:
            print('WARNING: could not read file:', path)
            return None
        vol_list.append(fin.load().astype(np.float32)[0, ])  # Extract the sole channel
    if len(vol_list) != len(file_tags):
        print('WARNING: did not read in correct number of files')
        return None
    return tuple(vol_list)
    
if __name__ == '__main__':
    pass
