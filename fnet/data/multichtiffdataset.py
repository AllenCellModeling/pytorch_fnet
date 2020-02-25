import numpy as np
import pandas as pd
import torch

from aicsimageio import AICSImage
from fnet.data.fnetdataset import FnetDataset


class MultiChTiffDataset(FnetDataset):
    """
    Dataset for multi-channel tiff files.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        path_csv: str = None,
        transform_signal=None,
        transform_target=None,
    ):

        super().__init__(dataframe, path_csv, transform_signal, transform_target)

        # if this column is a string assume it is in "[ind_1, ind_2, ..., ind_n]" format
        if isinstance(self.df["channel_signal"][0], str):
            self.df["channel_signal"] = [
                np.fromstring(ch[1:-1], sep=", ").astype(int)
                for ch in self.df["channel_signal"]
            ]
        else:
            self.df["channel_signal"] = [[int(ch)] for ch in self.df["channel_signal"]]

        if isinstance(self.df["channel_target"][0], str):
            self.df["channel_target"] = [
                np.fromstring(ch[1:-1], sep=", ").astype(int)
                for ch in self.df["channel_target"]
            ]
        else:
            self.df["channel_target"] = [[int(ch)] for ch in self.df["channel_target"]]

        assert all(
            i in self.df.columns
            for i in ["path_tiff", "channel_signal", "channel_target"]
        )

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: integer

        Returns
        -------
        C by <spatial dimensions> torch.Tensor
        """

        element = self.df.iloc[index, :]
        has_target = not np.any(np.isnan(element["channel_target"]))

        # aicsimageio.imread loads as STCZYX, so we load only CZYX
        with AICSImage(element["path_tiff"]) as img:
            im_tmp = img.get_image_data("CZYX", S=0, T=0)

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
        # im_out = [torch.unsqueeze(im, 0) for im in im_out]

        return tuple(im_out)

    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()
