from aicsimageio import imread
import pandas as pd
import numpy as np
import torch

from fnet.data.fnetdataset import FnetDataset


class MultiChTiffDataset(FnetDataset):
    """Dataset for multi-channel tiff files.

    Currently assumes that images are loaded in STCZYX format

    """

    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        path_csv: str = None,
        transform_signal=None,
        transform_target=None,
    ):

        super().__init__(dataframe, path_csv, transform_signal, transform_target)

        self.df["channel_signal"] = [int(ch) for ch in self.df["channel_signal"]]
        self.df["channel_target"] = [int(ch) for ch in self.df["channel_target"]]

        assert all(
            i in self.df.columns
            for i in ["path_tiff", "channel_signal", "channel_target"]
        )

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        has_target = not np.isnan(element["channel_target"])

        # aicsimageio.imread loads as STCZYX, so we load only CZYX
        im_tmp = imread(element["path_tiff"])[0, 0]

        im_out = list()
        im_out.append(im_tmp[element["channel_signal"]])

        if has_target:
            im_out.append(im_tmp[element["channel_target"]])

        if self.transform_signal is not None:
            for t in self.transform_signal:
                im_out[0] = t(im_out[0])

        if has_target and self.transform_target is not None:
            for t in self.transform_target:
                im_out[1] = t(im_out[1])

        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]

        # unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]

        return tuple(im_out)

    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()
