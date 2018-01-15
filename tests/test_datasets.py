import unittest

import torch

from fnet.data.czidataset import CziDataset
from fnet.data.tiffdataset import TiffDataset

import pandas as pd
import numpy as np
import pdb


class TestDataset(unittest.TestCase):
    def setUp(self):
        pass
        

    def data_format_verifier(self, datum):
        self.assertEqual(len(datum), 2)
        
        for channel in datum:
            self.assertIsInstance(channel, torch.Tensor)

    
    def test_czidataset(self):
        df_czi = pd.DataFrame.from_dict({'path_czi': ['../data/3500000883_100X_20170509_F08_P29.czi'], 
                                    'channel_signal': [0], 
                                    'channel_target': [1]})
        
        dataset = CziDataset(df_czi)
        
        datum = dataset[0]
        
        self.data_format_verifier(datum)
        
        
    def test_tiffdataset(self):
        
        df_tiff = pd.DataFrame.from_dict({'path_signal': ['../data/IF_to_EM_testdata/EM_low.tif'], 
                                        'path_target': ['../data/IF_to_EM_testdata/MBP_low.tif']})
        
        dataset = TiffDataset(df_tiff)
        
        datum = dataset[0]
        
        self.data_format_verifier(datum)        
        

if __name__ == '__main__':
    unittest.main()
