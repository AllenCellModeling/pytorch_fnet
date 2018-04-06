import torch.utils.data
from fnet.data.czireader import CziReader
from fnet.data.fnetdataset import FnetDataset
import pandas as pd
import numpy as np

import pdb

import fnet.transforms as transforms

class CziDataset(FnetDataset):
    """Dataset for CZI files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.normalize],
                    transform_target = None):
        
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
            
        self.transform_source = transform_source
        self.transform_target = transform_target
        
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        has_target = not np.isnan(element['channel_target'])
        czi = CziReader(element['path_czi'])
        
        im_out = list()
        im_out.append(czi.get_volume(element['channel_signal']))
        if has_target:
            im_out.append(czi.get_volume(element['channel_target']))
        
        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if has_target and self.transform_target is not None:
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])

                 
                 
        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        

        
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()
