import os

class DataSet(object):
    def __init__(self, path, percent_test=0.1):
        """
        path - path to parent directory that contains folders of data.
        percent_test - (float between 0.0 and 1.0) percent of folders to be used as test set
        """
        self._path = path
        self._percent_test = percent_test
        
        self._all_set = [i.path for i in os.scandir(self._path) if i.is_dir()]  # order is arbitrary
        # TODO: shuffle self._all_set
        idx_split = round(len(self._all_set)*percent_test)
        if idx_split == 0:
            idx_split = 1
        self._test_set = self._all_set[:idx_split]
        if len(self._all_set) == 1:
            print('WARNING: DataSet has only one element')
            idx_split = -1
        self._train_set = self._all_set[idx_split:]

    def get_test_set(self):
        return self._test_set
    
    def get_train_set(self):
        return self._train_set

    def _validate_folders(self):
        # TODO: validate all folders
        pass

    def __str__(self):
        str_list = []
        str_list.append('DataSet from: ' + self._path)
        str_list.append('percent_test: ' + str(self._percent_test))
        str_list.append('train/test/total: {:d}/{:d}/{:d}'.format(len(self._train_set),
                                                                  len(self._test_set),
                                                                  len(self._all_set)))
        return os.linesep.join(str_list)

def test():
    print('testing DataSet')
    data_path = '../data'
    percent_test = 0.2
    dataset = DataSet(data_path, percent_test=percent_test)
    print(dataset)
    test_set = dataset.get_test_set()
    train_set = dataset.get_train_set()
    test_tmp = set(test_set)
    train_tmp = set(train_set)
    union = test_tmp | train_tmp
    intersect = test_tmp & train_tmp
    print('union of test and train:', len(union))
    print('intersection of test and train:', len(intersect))
    assert len(union) == (len(test_set) + len(train_set))
    assert len(intersect) == 0
    print('*** TEST PASSED ***')

if __name__ == '__main__':
    test()
    
