import torch.utils.data
from fnet.data.czireader import CziReader
import pandas as pd

import pdb

import torchvision.transforms as transforms

class CziDataset(torch.utils.data.Dataset):
    """Dataset for CZI files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, transform = None):
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
            
        if transform is None:
            transform = transforms.ToTensor()
        
        self.transform = transform
            
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        czi = CziReader(element['path_czi'])
        
        im_out = (czi.get_volume(element['channel_signal']), czi.get_volume(element['channel_target']))
                  
        im_out = [self.transform(im.astype(float)) for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)
