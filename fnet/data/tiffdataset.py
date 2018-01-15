import torch.utils.data
from fnet.fnetdataset import FnetDataset
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
        tif_signal = TifReader(element['path_signal'])
	tif_target = TifReader(element['path_target'])
        return (
            tif_signal.get_image(element['path_signal']),
            tif_target.get_image(element['path_target'])
        )
    
    def __len__(self):
	return len(self.df)
