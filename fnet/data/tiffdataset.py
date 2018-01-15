import torch.utils.data
from fnet.data.fnetdataset import FnetDataset
from fnet.data.tifreader import TifReader
import pandas as pd

class TiffDataset(FnetDataset):
    """Dataset for Tif files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None,):
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        assert all(i in self.df.columns for i in ['path_signal', 'path_target'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        
        signal = TifReader(element['path_signal']).get_image()
        target = TifReader(element['path_target']).get_image()
        
        im_out = (signal, target)
        
        im_out = [torch.from_numpy(im).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)
