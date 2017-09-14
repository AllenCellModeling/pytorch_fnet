import os
import pickle
import glob
from aicsimage.io import omeTifReader
import numpy as np
from util import get_vol_transformed
import pandas as pd
import pdb
import warnings

BAN_LIST = (
    '3500000416_100X_20170117_E08_P15.czi',  # z-dim too small
    '3500000416_100X_20170117_E08_P16.czi',  # z-dim too small
    '3500000418_100X_20170117_F06_P16.czi',  # z-dim too small
    '3500000429_100X_20170120_F05_P36.czi',  # z-dim too small
    '3500000418_100X_20170117_F06_P18.czi',  # z-dim too small

    '3500000510_100X_20170131_E05_P07.czi',  # blank z-slices
    '3500000510_100X_20170131_E05_P08.czi',  # blank z-slices
)

class DataSet2(object):
    cell_lines = ('Tom20', 'Alpha tubulin', 'Sec61 beta', 'Alpha actinin',
                   'Desmoplakin', 'Lamin B1', 'Fibrillarin', 'Beta actin', 'ZO1',
                   'Myosin IIB')
    valid_chans = cell_lines + ('trans', 'dna', 'memb')
    
    def __init__(self,
                 path_load=None,
                 path_save=None,
                 path_csv=None,
                 train_select=True,
                 task=None,
                 chan=None,
                 train_set_size=None, percent_test=None,
                 transforms=None):
        """Load or build/save new data set.

        path_load - path to saved DataSet
        path_save - path where generated DataSet will be saved
        path_csv - path to csv that contains file paths
        train_select - (bool) if True, DataSet returns elements from training set, otherwise returns elements from test set
        task - {'ttf', 'snm'}
        chan - (str) target channel/protein from TTF; source channel/protein for SNM
        percent_test - (float between 0.0 and 1.0) percent of folders to be used as test set
        train_set_size - (int) number of elements in training set. If set, this will be used instead of percent_test.
        transforms - list/tuple of transforms, where each element is a transform or transform list to be applied
                     to a component of a DataSet element
        """
        assert (path_save is None) != (path_load is None), "must choose between either saving or loading a dataset"
        if path_load is None:
            assert path_csv.endswith('.csv')
            assert os.path.isfile(path_csv)
            assert chan in DataSet2.valid_chans
        
        self._path_save = path_load if path_load is not None else path_save
        self._path_csv = path_csv
        self._train_select = train_select
        self._task = task
        self._chan = chan
        self._percent_test = percent_test
        self._train_set_size = train_set_size
        self._transforms = transforms
        self._df_active = None
        self._df_csv = None
        
        if path_load is None:
            self._build_new_sets()
            self._save()
        else:
            self._load()
        if self._train_select:
            self._df_active = self._df_train
        else:
            self._df_active = self._df_test
        self._validate_dataset()

    def use_train_set(self):
        self._train_select = True
        self._df_active = self._df_train
        
    def use_test_set(self):
        self._train_select = False
        self._df_active = self._df_test

    def _get_state(self):
        """Returns a dict representing the DataSet."""
        state = {
            '_chan': self._chan,
            '_df_test': self._df_test,
            '_df_train': self._df_train,
            '_path_csv': self._path_csv,
            '_path_save': self._path_save,
            '_task': self._task,
            '_transforms': self._transforms
        }
        return state

    def _set_state(self, state):
        """Sets the DataSet state."""
        assert isinstance(state, dict)
        vars(self).update(state)

    def get_name(self, i, element_num):
        """Returns a name representing element i."""
        name = os.path.basename(self._df_active.iloc[i, element_num])
        return name

    def get_df_csv(self):
        return self._df_csv

    def _build_new_sets(self):
        """Create test_set and train_set instance variables."""
        print('reading:', self._path_csv)
        df_csv = pd.read_csv(self._path_csv)
        self._df_csv = df_csv
        if self._task == 'ttf':
            if self._chan in DataSet2.cell_lines:
                columns = ('save_trans_path', 'save_struct_path')
                mask = df_csv['structureProteinName'] == self._chan
                # mask = mask & df_csv['inputFolder'].str.contains('aics/microscopy') # look at only images from microscopy team
                for banned in BAN_LIST:
                    mask = mask & ~df_csv['inputFilename'].str.contains(banned)
                df_all = df_csv.loc[mask, columns]
            elif self._chan in ('dna', 'memb'):
                mask = df_csv['inputFolder'].str.contains('aics/microscopy')
                warnings.warn('skipping bad plate: 3500000926')
                mask = mask & ~df_csv['inputFilename'].str.contains('3500000926')  # bad plate
                columns = ('save_trans_path', 'save_{}_path'.format(self._chan))
                df_all = df_csv.loc[mask, columns]
            else:
                raise AttributeError
        else:
            raise NotImplementedError
        df_all = df_all.sample(frac=1)
        
        if df_all.shape[0] == 1:
            warnings.warn('DataSet has only one element. Training and test sets will be identical.')
            self._df_test = df_all
            self._df_train = df_all
            return
        if self._train_set_size is not None:
            if self._train_set_size == 0:
                self._df_test = df_all
                self._df_train = df_all[0:0]  # empty DataFram but with columns intact
                return
            idx_split = self._train_set_size*-1
        else:
            idx_split = round(len(df_all)*self._percent_test)
        self._df_test = df_all[:idx_split]
        self._df_train = df_all[idx_split:]
    
    def get_active_set(self):
        return self._df_active
    
    def get_test_set(self):
        return self._df_test
    
    def get_train_set(self):
        return self._df_train

    def _validate_dataset(self):
        assert self._df_active is not None
        if self._transforms is not None:
            assert len(self._df_active.columns) == len(self._transforms)
        for col in self._df_active.columns:
            assert self._df_active[col].nunique() == self._df_active.shape[0], 'paths should be unique'
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
        if id(self._df_train) == id(self._df_test):
            n_unique = self._df_train.shape[0]
        else:
            n_unique = self._df_train.shape[0] + self._df_test.shape[0]
        str_active = 'train' if self._train_select else 'test'
        str_list = []
        str_list.append('{} from: {}'.format(self.__class__.__name__, self._path_save))
        str_list.append('path_csv: ' + str(self._path_csv))
        str_list.append('active_set: ' + str_active)
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._df_train),
                                                                  len(self._df_test),
                                                                  n_unique))
        str_list.append('transforms: ' + get_str_transform(self._transforms))
        return os.linesep.join(str_list)

    def get_item_sel(self, idx, sel, apply_transforms=True):
        """Get sel-th item from dataset element idx."""
        path_pre = self._df_active.iloc[idx, sel]
        base = os.path.dirname(self._path_csv)
        path = os.path.join(base, path_pre)
        try:
            print('reading:', path)
            fin = omeTifReader.OmeTifReader(path)
            volume_pre = fin.load().astype(np.float32)[0, ]  # Extract the sole channel
            fin.close()
        except:
            warnings.warn('could not read file: {}'.format(path))
            return None
        if self._transforms is None or not apply_transforms:
            volume = volume_pre
        else:
            volume = get_vol_transformed(volume_pre, self._transforms[sel])
        return volume

    def __len__(self):
        return len(self._df_active)

    def __getitem__(self, index):
        """Returns arrays corresponding to files identified by file_tags in the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - n-element tuple or None. If the file read was successful, return tuple
                  of transformed arrays else return None
        """
        paths_pre = self._df_active.iloc[index]
        base = os.path.dirname(self._path_csv)
        paths = (os.path.join(base, path_pre) for path_pre in paths_pre)

        volumes_pre = []
        for path in paths:
            try:
                print('reading:', path)
                fin = omeTifReader.OmeTifReader(path)
                volumes_pre.append(fin.load().astype(np.float32)[0, ])  # Extract the sole channel
                fin.close()
            except:
                warnings.warn('could not read file: {}'.format(path))
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
    file_list = [i.path for i in os.scandir(path_dir) if i.is_file()]  # order is arbitrary

    paths_to_read = []
    for tag in file_tags:
        matches = [f for f in file_list if (tag in f)]
        if len(matches) != 1:
            warnings.warn('incorrectect number of files found for pattern {} in {}'.format(bit, path_dir))
            return None
        paths_to_read.append(matches[0])
    
    vol_list = []
    for path in paths_to_read:
        fin = omeTifReader.OmeTifReader(path)
        try:
            print('reading:', path)
            fin = omeTifReader.OmeTifReader(path)
        except:
            warnings.warn('could not read file: {}'.format(path))
            return None
        vol_list.append(fin.load().astype(np.float32)[0, ])  # Extract the sole channel
        fin.close()
    if len(vol_list) != len(file_tags):
        warnings.warn('did not read in correct number of files')
        return None
    return tuple(vol_list)
    
DataSet = DataSet2

if __name__ == '__main__':
    pass
