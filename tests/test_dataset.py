import unittest
import util.data
import util.data.transforms
import numpy as np

class TestDataSet(unittest.TestCase):
    def test_builtandload(self):
        path = 'data/few_files'
        train_select = True
        file_tags = ('_trans.tif', '_dna.tif', '_trans.tif')
        train_set_size = 4
        percent_test = 1.0
        # aiming for 0.3 um/px
        z_fac = 0.97
        xy_fac = 0.36
        resize_factors = (z_fac, xy_fac, xy_fac)
        # signal_transforms = (util.data.transforms.Resizer(resize_factors), util.data.transforms.sub_mean_norm)
        signal_transforms = util.data.transforms.Resizer(resize_factors)
        target_transforms = (util.data.transforms.Resizer(resize_factors), util.data.transforms.do_nothing)
        transforms = (signal_transforms, target_transforms, None)
        dataset = util.data.DataSet(path, file_tags=file_tags,
                                    train_select=train_select, percent_test=percent_test, train_set_size=train_set_size,
                                    transforms=transforms)

        dataset_2 = util.data.DataSet(path, train_select=False)
        data = dataset_2[0]
        self.assertEqual(len(data), len(file_tags))
        self.assertTrue(isinstance(data[0], np.ndarray))
        shape_exp = tuple(round(data[-1].shape[i]*resize_factors[i]) for i in range(3))
        self.assertEqual(data[0].shape, shape_exp)
        self.assertEqual(data[1].shape, shape_exp)
        n_dirs = len(dataset_2.get_test_set()) + len(dataset_2.get_train_set())
        self.assertEqual(len(dataset_2), n_dirs - train_set_size)
