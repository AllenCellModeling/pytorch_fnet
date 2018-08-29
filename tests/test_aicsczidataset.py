from fnet.data.aicsczidataset import AICSCziDataset
import fnet.transforms as transforms
import numpy as np
import os
import pandas as pd
import pdb
import shutil
import torch
import tifffile
import unittest

class TestTransform:
    def __call__(self, ar):
        low, hi = np.percentile(ar, [1, 99])
        print('low, hi', low, hi)
        ar = ar - low
        ar = ar/(hi - low)*256.0
        ar[ar < 0] = 0
        ar[ar > 255.0] = 255.0
        return ar.astype(np.uint8)

    def __repr__(self):
        return 'TestTransform(2, 3, 6)'

class TestTransform2:
    def __call__(self, ar):
        return ar

    def __repr__(self):
        return 'TestTransform2()'

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_czi = os.path.join(os.path.dirname(__file__), '..', 'data', '3500000427_100X_20170120_F05_P27.czi')


    def test_zlimits(self):
        df = pd.DataFrame({
            'path_czi': [self.path_czi, self.path_czi, self.path_czi],
            'channel_signal': 0,
            'channel_target': 1,
            'zlim_min': [None, 33, None],
            'zlim_max': [None, 37, 20],
            'xlim_min': [None, None, 3]
        })
        transforms = ["fnet.transforms.Normalize(0)"]
        ds = AICSCziDataset(
            df,
            transform_signal = transforms,
            transform_target = transforms,
        )
        shapes_exp = [
            (1, 39, 512, 512),
            (1, 5, 512, 512),
            (1, 21, 512, 509),
        ]
        for idx_data, data in enumerate(ds):
            self.assertEqual(tuple(data[0].size()), shapes_exp[idx_data])
            self.assertEqual(tuple(data[1].size()), shapes_exp[idx_data])


    def test_caching(self):
        # Both signal and target channels specified
        path_cache_dir = os.path.join('tests', '.tmp')
        if os.path.exists(path_cache_dir):
            shutil.rmtree(path_cache_dir)
        df = pd.DataFrame({
            'path_czi': [self.path_czi],
            'channel_signal': 0,
            'channel_target': 1,
        })
        ds = AICSCziDataset(
            df,
            path_cache_dir = path_cache_dir,
            transform_target = [TestTransform(), TestTransform2()],
        )
        data = ds[0]
        self.assertEqual(len(data), 2)
        self.assertTrue(all([d.dtype == torch.float for d in data]))  # Check dtype
        self.assertTrue(all([len(d.size()) == 4 for d in data]))  # Check n dimensions
        
        # Check that the number of cached files is correct
        self.assertEqual(len([p.path for p in os.scandir(path_cache_dir)]), 2)

        # No caching
        ds = AICSCziDataset(df, transform_signal = [TestTransform2()])
        data = ds[0]
        self.assertEqual(len([p.path for p in os.scandir(path_cache_dir)]), 2)
        
        # Only signal channel specified
        df = pd.DataFrame({
            'path_czi': [self.path_czi],
            'channel_signal': 1,
            'channel_target': np.nan,
        })
        ds = AICSCziDataset(
            df,
            path_cache_dir = path_cache_dir,
            transform_signal = [TestTransform(), TestTransform2()],
        )
        data = ds[0]
        self.assertEqual(len(data), 1)
        self.assertEqual(len([p.path for p in os.scandir(path_cache_dir)]), 2)
        
        shutil.rmtree(path_cache_dir)

    def test_augmentations(self):
        path_cache_dir = os.path.join('tests', '.tmp')
        if not os.path.exists(path_cache_dir):
            os.makedirs(path_cache_dir)
        
        df = pd.DataFrame({
            'path_czi': [self.path_czi]*4,
            'channel_signal': 3,
            'channel_target': 2,
            'flip_y': [None, 1, 0, 1],
            'flip_x': [None, 0, 1, 1],
        })
        print(df)
        ds = AICSCziDataset(
            df,
            path_cache_dir = path_cache_dir,
            transform_signal = [TestTransform(), TestTransform2()],
            transform_target = [TestTransform(), TestTransform2()],
        )
        for idx_ds in range(len(ds)):
            print('idx:', idx_ds)
            data = ds[idx_ds]
            for idx_data, d in enumerate(data):
                path_save = os.path.join(path_cache_dir, f'{idx_ds}_{idx_data}.tiff')
                tifffile.imsave(path_save, d.numpy())
                print(d.shape, path_save)
        

if __name__ == '__main__':
    unittest.main()
