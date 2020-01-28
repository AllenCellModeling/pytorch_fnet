import os

import numpy as np
import pandas as pd
import tifffile
import torch

from fnet.data.tiffdataset import TiffDataset
from fnet.utils.general_utils import add_augmentations


def dummy_fnet_dataset(train: bool = False) -> TiffDataset:
    """Returns a dummy Fnetdataset."""
    df = pd.DataFrame(
        {
            "path_signal": [os.path.join("data", "EM_low.tif")],
            "path_target": [os.path.join("data", "MBP_low.tif")],
        }
    ).rename_axis("arbitrary")
    if not train:
        df = add_augmentations(df)
    return TiffDataset(dataframe=df)


class _CustomDataset:
    """Custom, non-FnetDataset."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        loc = self._df.index[idx]
        sig = torch.from_numpy(
            tifffile.imread(self._df.loc[loc, "path_signal"])[np.newaxis,]
        )
        tar = torch.from_numpy(
            tifffile.imread(self._df.loc[loc, "path_target"])[np.newaxis,]
        )
        return (sig, tar)


def dummy_custom_dataset(train: bool = False) -> TiffDataset:
    """Returns a dummy custom dataset."""
    df = pd.DataFrame(
        {
            "path_signal": [os.path.join("data", "EM_low.tif")],
            "path_target": [os.path.join("data", "MBP_low.tif")],
        }
    )
    if not train:
        df = add_augmentations(df)
    return _CustomDataset(df)
