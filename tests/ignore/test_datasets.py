import unittest

import torch

from fnet.data.czidataset import CziDataset
from fnet.data.tiffdataset import TiffDataset

import fnet.transforms as transforms

import pandas as pd
import numpy as np
import pdb


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.df_czi = pd.DataFrame.from_dict({'path_czi': ['../data/3500000427_100X_20170120_F05_P27.czi'], 
                                    'channel_signal': [0], 
                                    'channel_target': [1]})
        
        self.df_tiff = pd.DataFrame.from_dict({'path_signal': ['../data/EM_low.tif'], 
                                'path_target': ['../data/MBP_low.tif']})
        
        
    def test_czidataset(self):
        self.dataset_verifier(self.df_czi, CziDataset)
                                                                     
    def test_tiffdataset(self):
        self.dataset_verifier(self.df_tiff, TiffDataset)

    
        
    def dataset_verifier(self, df, dataset_object):
        #Run all tests for a particular dataframe/dataset pair
        self.data_format_verifier(df, dataset_object)
        self.data_normalize_verifier(df, dataset_object)
        self.data_resize_verifier(df, dataset_object)

    def data_format_verifier(self, df, dataset_object):
        #Make sure the dataset resturns a pytorch tensor (Float by default)
        dataset = dataset_object(df)
        datum = dataset[0]
        
        self.assertEqual(len(datum), 2)
        
        for channel in datum:
            self.assertIsInstance(channel, torch.Tensor)

    def data_normalize_verifier(self, df, dataset_object):
        #Make sure the normalization transform works for both source and target
        dataset = dataset_object(df, transform_source=[transforms.normalize], transform_target = None )
        datum = dataset[0]
        
        self.assertAlmostEqual(torch.mean(datum[0]), 0)
        self.assertNotAlmostEqual(torch.mean(datum[1]), 0)
    
        #Normalize target, not source
        dataset = dataset_object(df, transform_source = None, transform_target=[transforms.normalize] )
        
        datum = dataset[0]
        
        self.assertNotAlmostEqual(torch.mean(datum[0]), 0)
        self.assertAlmostEqual(torch.mean(datum[1]), 0)
        
    def data_resize_verifier(self, df, dataset_object):
        #Make sure resizing the image works (for 2D or 3D cases)
        resize_xy = 0.3
        resize_z = 1
        
        #Get normal-sized dataset
        dataset = dataset_object(df, transform_source = None, transform_target = None)
        datum = dataset[0]
        datum_size = [d for d in datum[0].size()[1:]]
        
        if len(datum_size) == 2:
            resizer = transforms.Resizer([resize_xy, resize_xy])
        elif len(datum_size) == 3:
            resizer = transforms.Resizer([resize_z, resize_xy, resize_xy])
        else:
            print("Data must be 2 or 3 dimensions", sys.exc_info()[0])
            raise
        
        
        #Get resized dataset
        dataset_resize = dataset_object(df, transform_source = [resizer], transform_target = None)
        datum_resize = dataset_resize[0]
        datum_resize_size = [d for d in datum_resize[0].size()[1:]]
        
        c = 0
        if len(datum_size) == 3:
            self.assertEqual(datum_resize_size[c], round(datum_size[c]*resize_z))
            c+=1
               
        self.assertEqual(datum_resize_size[c], round(datum_size[c]*resize_xy))
        c+=1
        self.assertEqual(datum_resize_size[c], round(datum_size[c]*resize_xy))
    
         

if __name__ == '__main__':
    unittest.main()
