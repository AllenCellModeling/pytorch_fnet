import os
import pickle
import glob
from aicsimage.io import omeTifReader
import numpy as np
from natsort import natsorted
from util import get_vol_transformed
import pdb
import warnings

class DataSet(object):
    def __init__(self,
                 path_load=None,
                 path_save=None,
                 path_source=None,
                 train_select=True,
                 file_tags=None,
                 train_set_size=None, percent_test=None,
                 transforms=None):
        """Load or build/save new data set.

        path_load - path to saved DataSet
        path_save - path where generated DataSet will be saved
        path_source - path to parent directory that contains folders of data
        train_select - (bool) if True, DataSet returns elements from training set, otherwise returns elements from test set
        file_tags - (list or tuple or strings) string "tags" that can uniquely identify files in the directory specified by path
        percent_test - (float between 0.0 and 1.0) percent of folders to be used as test set
        train_set_size - (int) number of elements in training set. If set, this will be used instead of percent_test.
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        assert (path_save is None) != (path_load is None), "must choose between either saving or loading a dataset"
        self._path_save = path_load if path_load is not None else path_save
        self._path_source = path_source
        self._file_tags = file_tags
        self._train_select = train_select
        self._percent_test = percent_test
        self._train_set_size = train_set_size
        self._transforms = transforms
        self._active_set = None
        
        if path_load is None:
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

    def _get_state(self):
        """Returns a dict representing the DataSet."""
        state = {
            '_file_tags': self._file_tags,
            '_path_save': self._path_save,
            '_path_source': self._path_source,
            '_test_set': self._test_set,
            '_train_set': self._train_set,
            '_transforms': self._transforms
        }
        return state

    def _set_state(self, state):
        """Sets the DataSet state."""
        assert isinstance(state, dict)
        vars(self).update(state)

    def get_name(self, i):
        """Returns a name representing element i."""
        name = os.path.basename(self._active_set[i])
        return name

    def _build_new_sets(self):
        """Create test_set and train_set instance variables."""
        all_set = [i.path for i in os.scandir(self._path_source) if i.is_dir()]  # order is arbitrary
        # TODO: shuffle _all_set?
        if len(all_set) == 1:
            warnings.warn('DataSet has only one element. Training and test sets will be identical.')
            self._test_set = all_set
            self._train_set = all_set
            return
        if self._train_set_size is not None:
            if self._train_set_size == 0:
                self._test_set = all_set[:]
                self._train_set = []
                return
            idx_split = self._train_set_size*-1
        else:
            idx_split = round(len(all_set)*self._percent_test)
        self._test_set = all_set[:idx_split]
        self._train_set = all_set[idx_split:]
    
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
        dirname = os.path.dirname(self._path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        package = self._get_state()
        with open(self._path_save, 'wb') as fo:
            pickle.dump(package, fo)
            print('saved dataset to:', self._path_save)

    def _load(self):
        with open(self._path_save, 'rb') as fin:
            package = pickle.load(fin)
        print('loaded dataset from:', self._path_save)
        self._set_state(package)

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
        n_unique = len(set(self._train_set) | set(self._test_set))
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('{} from: {}'.format(self.__class__.__name__, self._path_save))
        str_list.append('file_tags: ' + str(self._file_tags))
        str_list.append('active_set: ' + str_active)
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._train_set),
                                                                  len(self._test_set),
                                                                  n_unique))
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
            if self._transforms is None:
                volumes.append(volume_pre)
            else:
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
        fin.close()
    if len(vol_list) != len(file_tags):
        print('WARNING: did not read in correct number of files')
        return None
    return tuple(vol_list)
    
if __name__ == '__main__':
    pass
