import torch.utils.data
from fnet.data.fnetdataset import FnetDataset
from fnet.data.tifreader import TifReader

import fnet.transforms as transforms

import pandas as pd

import pdb

class TiffDataset(FnetDataset):
    """Dataset for Tif files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.normalize],
                    transform_target = None):
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        assert all(i in self.df.columns for i in ['path_signal', 'path_target'])
        
        self.transform_source = transform_source
        self.transform_target = transform_target

    def __getitem__(self, index):
        element = self.df.iloc[index, :]

        im_out = [TifReader(element['path_signal']).get_image()]
        if isinstance(element['path_target'], str):
            im_out.append(TifReader(element['path_target']).get_image())
        
        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if self.transform_target is not None and (len(im_out) > 1):
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])


        
        im_out = [torch.from_numpy(im).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index):
        return self.df.iloc[index, :].to_dict()
