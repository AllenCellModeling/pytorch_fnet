import unittest
import util.data
import util.data.transforms
import numpy as np
import pdb

class TestDataSet(unittest.TestCase):
    def test_builtandload(self):
        path = 'data/few_files'
        file_tags = ('_dna.tif', '_nuc.tif', '_cell.tif')
        train_select = True
        percent_test = 0.1
        train_set_size = 4
        # aiming for 0.3 um/px
        z_fac = 0.97
        xy_fac = 0.36
        resize_factors = (z_fac, xy_fac, xy_fac)
        resizer = util.data.transforms.Resizer(resize_factors)
        transforms = (resizer, resizer, resizer)
        dataset = util.data.NucMaskDataSet(path, file_tags=file_tags,
                                           train_select=train_select, percent_test=percent_test, train_set_size=train_set_size,
                                           transforms=transforms)
        # load saved DataSet
        dataset_2 = util.data.NucMaskDataSet(path, train_select=False)
        data = dataset_2[0]
        self.assertEqual(len(data), 2)
        self.assertTrue(isinstance(data[0], np.ndarray))
        self.assertTrue(isinstance(data[1], np.ndarray))
        self.assertEqual(data[0].shape, data[1].shape)
        self.assertGreaterEqual(np.min(data[1]), 0)
        self.assertLessEqual(np.max(data[1]), 2)
        n_dirs = len(dataset_2.get_test_set()) + len(dataset_2.get_train_set())
        self.assertEqual(len(dataset_2), n_dirs - train_set_size)
