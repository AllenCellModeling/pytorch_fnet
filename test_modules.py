from util.DataSet import DataSet

def test_DataSet():
    print('testing DataSet')
    path = 'data/few_files'
    dataset = DataSet(path, percent_test=0.1)
    print(dataset)
    print(dataset.get_test_set())
    print(dataset.get_train_set())
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
    test_DataSet()
