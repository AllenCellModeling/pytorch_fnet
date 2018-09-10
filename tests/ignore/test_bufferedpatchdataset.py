from fnet.data import CziDataset
import fnet.transforms
import os
import pandas as pd
import pdb
import torch
import unittest

class TestBufferedPatchDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path_dirname_test = os.path.dirname(__file__)
        path_test_czi = os.path.join(path_dirname_test, '..', 'data', '3500000427_100X_20170120_F05_P27.czi')
        print(path_test_czi)
        df_dataset = pd.DataFrame({
                'path_czi': [path_test_czi],
                'channel_signal':3,
                'channel_target':2,
            })
        transforms = [fnet.transforms.do_nothing]
        cls.dataset = CziDataset(
            df_dataset,
            transform_source = transforms,
            transform_target = transforms,
        )
        
    def test_2d_from_3d(self):
        # expected datum shape: 1x39x512x512
        patch_size = [1, 32, 128]
        bpds_no_squeeze = fnet.data.BufferedPatchDataset(
            dataset = self.dataset,
            patch_size = patch_size,
            buffer_size = 1,
            buffer_switch_frequency = -1,
            npatches = 16,
            verbose = True,
        )
        patches_no_squeeze = bpds_no_squeeze[0]
        print(patches_no_squeeze[0].size())
        self.assertEqual(tuple(patches_no_squeeze[0].size()), (1,) + tuple(patch_size))
        self.assertEqual(tuple(patches_no_squeeze[1].size()), (1,) + tuple(patch_size))
        
        bpds = fnet.data.BufferedPatchDataset(
            dataset = self.dataset,
            patch_size = patch_size,
            buffer_size = 1,
            buffer_switch_frequency = -1,
            npatches = 16,
            verbose = True,
            dim_squeeze = 1,  # turn 1-channel 3d, patch (zdim 1) into 1-channel, 2d patch
        )
        patches = bpds[0]
        print(patches[0].size())
        self.assertEqual(tuple(patches[0].size()), tuple(patch_size))
        self.assertEqual(tuple(patches[1].size()), tuple(patch_size))

    def test_sample_entire_range(self):
        # Create 1-element dataset with signal/target of size (1, 8, 10)
        len_dim_test = 8
        size_patch = [1, 1]

        for dim in range(2):        
            dataset_x = (torch.zeros(len_dim_test, len_dim_test) + torch.arange(len_dim_test)).int()
            if dim == 0:
                dataset_x = torch.transpose(dataset_x, 0, 1)
            dataset_x = torch.unsqueeze(dataset_x, 0)
            dataset = ((dataset_x, dataset_x*-1),)
            bpds = fnet.data.BufferedPatchDataset(
                dataset = dataset,
                patch_size = size_patch,
                buffer_size = 1,
                buffer_switch_frequency = -1,
                npatches = 16,
                verbose = True,
            )
            seen = set()
            for idx in range(64):
                signal, target = bpds[idx]
                seen.update(signal[0, 0, ])
            self.assertEqual(seen, set(range(len_dim_test)))  # check if all indices along dimension have been sampled at least once
