import unittest
import util.data
import util.data.transforms

class TestDataSet(unittest.TestCase):

    def test_simple(self):
        path = 'data/few_files'
        
        train_select = True
        train_set_limit = None
        percent_test = 0.1

        z_fac = 0.97
        xy_fac = 0.36
        resize_factors = (z_fac, xy_fac, xy_fac)
        
        transform = util.data.transforms.Resizer(resize_factors)
        target_transform = util.data.transforms.Resizer(resize_factors)
        
        dataset = util.data.DataSet(path, train=train_select,
                                    percent_test=percent_test, train_set_limit=train_set_limit,
                                    transform=transform,
                                    target_transform=target_transform,
                                    force_rebuild=True)
        print(dataset)
        test_set = dataset.get_test_set()
        train_set = dataset.get_train_set()
        test_tmp = set(test_set)
        train_tmp = set(train_set)
        union = test_tmp | train_tmp
        intersect = test_tmp & train_tmp
        self.assertEqual(len(union), len(test_set) + len(train_set))
        self.assertEqual(len(intersect), 0)
        
        if train_select:
            self.assertEqual(len(dataset), len(train_set))
        else:
            self.assertEqual(len(dataset), len(test_set))
        tmp = dataset[0]
        print('tmp:', tmp[0].shape, tmp[1].shape)
