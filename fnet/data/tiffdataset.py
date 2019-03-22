from fnet.data.fnetdataset import FnetDataset
from fnet.utils.general_utils import add_augmentations
from typing import Optional
import numpy as np
import tifffile
import torch


def _flip_y(ar):
    """Flip array along y axis.

    Array should have dimensions ZYX or YX.

    """
    return np.flip(ar, axis=-2)


def _flip_x(ar):
    """Flip array along x axis.

    Array should have dimensions ZYX or YX.

    """
    return np.flip(ar, axis=-1)


class TiffDataset(FnetDataset):
    """Dataset where each row is a signal-target pairing from TIFF files.

    Dataset items will be 2-item tuples:
        (signal image, target image)

    Parameters
    ----------
    augment
        Set to augment dataset with flips about the x and/or y axis.

    """

    def __init__(
            self,
            col_index: Optional[str] = None,
            col_signal: str = 'path_signal',
            col_target: str = 'path_target',
            augment: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert col_signal in self.df.columns
        assert col_target in self.df.columns
        self.col_index = col_index
        self.col_signal = col_signal
        self.col_target = col_target
        self.augment = augment
        if self.col_index is not None:
            self.df = self.df.set_index(self.col_index)
        if self.augment:
            self.df = add_augmentations(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        flip_y = self.df.iloc[idx, :].get('flip_y', -1) > 0
        flip_x = self.df.iloc[idx, :].get('flip_x', -1) > 0
        datum = []
        for col, transforms in [
                [self.col_signal, self.transform_signal],
                [self.col_target, self.transform_target],
        ]:
            path_read = self.df.loc[self.df.index[idx], col]
            if not isinstance(path_read, str):
                datum.append(None)
                continue
            ar = tifffile.imread(path_read)
            if transforms is None:
                transforms = []
            if flip_y:
                transforms.append(_flip_y)
            if flip_x:
                transforms.append(_flip_x)
            for transform in transforms:
                ar = transform(ar)
            datum.append(
                torch.tensor(ar[np.newaxis, ].copy(), dtype=torch.float32)
            )
        return tuple(datum)

    def get_information(self, idx: int) -> dict:
        """Returns information about the dataset item.

        Parameters
        ----------
        idx
            Index of dataset item for which to retrieve information.

        Returns
        -------
        dict
           Information about dataset item.

        """
        return self.df.loc[idx, :].to_dict()
