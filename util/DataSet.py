import os
import pickle

class DataSet(object):
    def __init__(self, path, percent_test=0.1, force_rebuild=False, train_set_limit=None):
        """
        path - path to parent directory that contains folders of data.
        percent_test - (float between 0.0 and 1.0) percent of folders to be used as test set
        """
        # member variables
        self._path = path
        self._percent_test = percent_test
        self._train_set_limit = train_set_limit
        self._all_set = None
        self._test_set = None
        self._train_set = None
        
        self._save_path = os.path.join(os.path.dirname(path), 'dataset_saves', os.path.basename(path) + '.p')
        if not os.path.exists(self._save_path) or force_rebuild:
            self._build_new_sets()
            self._save()
        else:
            self._load()

    def _build_new_sets(self):
        self._all_set = [i.path for i in os.scandir(self._path) if i.is_dir()]  # order is arbitrary
        # TODO: shuffle self._all_set
        if len(self._all_set) == 1:
            print('WARNING: DataSet has only one element')
            self._test_set = self._all_set
            self._train_set = self._all_set
        else:
            idx_split = round(len(self._all_set)*self._percent_test)
            self._test_set = self._all_set[:idx_split]
            self._train_set = self._all_set[idx_split:]
            if self._train_set_limit is not None:
                self._train_set = self._train_set[:self._train_set_limit]
                print('limiting training set to {:d} elements'.format(self._train_set_limit))
    
    def get_test_set(self):
        return self._test_set
    
    def get_train_set(self):
        return self._train_set

    def _validate_folders(self):
        # TODO: validate all folders
        pass

    def _save(self):
        dirname = os.path.dirname(self._save_path)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        package = (self._all_set, self._path, self._percent_test, self._test_set, self._train_set)
        assert len(package) == 5
        with open(self._save_path, 'wb') as fo:
            pickle.dump(package, fo)
            print('saved dataset split to:', self._save_path)

    def _load(self):
        with open(self._save_path, 'rb') as fin:
            package = pickle.load(fin)
        assert len(package) == 5
        (self._all_set, self._path, self._percent_test, self._test_set, self._train_set) = package
        print('loaded dataset split from:', self._save_path)

    def __str__(self):
        str_list = []
        str_list.append('DataSet from: ' + self._path)
        str_list.append('percent_test: ' + str(self._percent_test))
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._train_set),
                                                                  len(self._test_set),
                                                                  len(self._all_set)))
        return os.linesep.join(str_list)

if __name__ == '__main__':
    pass
    
