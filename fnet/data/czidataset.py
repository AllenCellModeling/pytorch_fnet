from fnet.data.czireader import CziReader
from fnet.data.fnetdataset import FnetDataset
import numpy as np
import pdb  # noqa: F401
import torch.utils.data


class CziDataset(FnetDataset):
    """Dataset for CZI files.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index):
        element = self.df.iloc[index, :]
        has_target = not np.isnan(element["channel_target"])
        czi = CziReader(element["path_czi"])

        im_out = list()
        im_out.append(czi.get_volume(element["channel_signal"]))
        if has_target:
            im_out.append(czi.get_volume(element["channel_target"]))
        if self.transform_signal is not None:
            for t in self.transform_signal:
                im_out[0] = t(im_out[0])
        if has_target and self.transform_target is not None:
            for t in self.transform_target:
                im_out[1] = t(im_out[1])
        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
        # unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        return im_out

    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()
