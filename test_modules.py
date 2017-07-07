import util.data

def test_DataSet():
    print('***** testing DataSet *****')
    # path = 'data/few_files'
    path = 'data/no_hots'
    train_select = True
    # aiming for 0.3 um/px
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    dataset = util.data.DataSet(path, train=train_select, force_rebuild=True, train_set_limit=5,
                                transform=util.data.transforms.Resizer(resize_factors))
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
    print('dataset length:', len(dataset))
    if train_select:
        assert len(dataset) == len(train_set)
    else:
        assert len(dataset) == len(test_set)
    tmp = dataset[0]
    print('tmp:', tmp[0].shape, tmp[1].shape)
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

def test_multi_file_data_provider():
    print('***** testing MultiFileDataProvider *****')
    path = 'data/one_file'
    dataset = util.data.DataSet(path, train=True)
    print(dataset)
    print()
    fifo_size = 3
    n_iter = 4
    dp = util.data.MultiFileDataProvider(dataset, fifo_size, n_iter, 4)
    for i, batch in dp:
        print('got batch')
    

if __name__ == '__main__':
    test_DataSet()
    # test_TiffDataProvider()
    # test_save_checkpoint()
    # test_load_checkpoint()
    # test_multi_file_data_provider()
