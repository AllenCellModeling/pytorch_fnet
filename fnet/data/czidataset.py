import torch.utils.data
from fnet.data.czireader import CziReader
import pandas as pd

class CziDataset(torch.utils.data.Dataset):
    """Dataset for CZI files."""

    def __init__(self, path_csv: str = None, dataframe: pd.DataFrame = None):
        if path_csv is not None:
            self.df = pd.read_csv(path_csv)
        else:
            self.df = dataframe
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        czi = CziReader(element['path_czi'])
        return (
            czi.get_volume(element['channel_signal']),
            czi.get_volume(element['channel_target'])
        )
    
    def __len__(self):
        return len(self.df)
