from util.DataSet import DataSet
from util.TiffDataProvider import TiffDataProvider
import model_modules.u_net_v0 as trainer

def test_DataSet():
    print('testing DataSet')
    # path = 'data/few_files'
    path = 'data/tubulin_nobgsub'
    dataset = DataSet(path, percent_test=0.1, force_rebuild=True, train_set_limit=50)
    print(dataset)
    # print(dataset.get_test_set())
    # print(dataset.get_train_set())
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

def test_TiffDataProvider():
    print('testing TiffDataProvder')
    path = 'data/test_files'
    dataset = DataSet(path, percent_test=0.0, force_rebuild=True)
    print(dataset)
    train_set = dataset.get_train_set()
    print(train_set)
    tiff_dp = TiffDataProvider(train_set, 3, 1)
    for batch in tiff_dp:
        stats = tiff_dp.get_last_batch_stats()
        print(stats)
    # TODO: change the way iterations are determined when encoutering a bad file

def test_save_checkpoint():
    model = trainer.Model(mult_chan=32, depth=4)
    model.meta['name'] = 'chek'
    print(model)
    path = 'test_dir/test_checkpoint.pk'
    model.save_checkpoint(path)
    
def test_load_checkpoint():
    path = 'test_dir/test_checkpoint.pk'
    model = trainer.Model(load_path=path)
    print(model)

if __name__ == '__main__':
    test_DataSet()
    # test_TiffDataProvider()
    # test_save_checkpoint()
    # test_load_checkpoint()
