from fnet.utils.general_utils import to_objects, whats_my_name
from typing import List, Optional, Union
import pandas as pd
import torch.utils.data


def _to_str_list(olist: List) -> Optional[List[str]]:
    """Turns a list of objects into a list of the objects' string
    representations.

    """
    if olist is None:
        return None
    return [whats_my_name(o) for o in olist]


class _LocIndexer:
    """'Loc' indexer of objects with a 'df' (DataFrame) attribute."""

    def __init__(self, super_obj):
        assert isinstance(super_obj.df, pd.DataFrame)
        self.super_obj = super_obj

    def __getitem__(self, idx):
        idx_trans = self.super_obj.df.index.get_loc(idx)
        return self.super_obj[idx_trans]


class _iLocIndexer:
    """'iLoc' indexer of objects with a 'df' (DataFrame) attribute."""

    def __init__(self, super_obj):
        assert isinstance(super_obj.df, pd.DataFrame)
        self.super_obj = super_obj

    def __getitem__(self, idx):
        return self.super_obj[idx]


class FnetDataset(torch.utils.data.Dataset):
    """Abstract class for fnet datasets.

    Parameters
    ----------
    dataframe
        DataFrame where rows are dataset elements. Overrides path_csv.
    path_csv
        Path to csv from which to create DataFrame.
    transform_signal
        List of transforms to apply to signal image.
    transform_target
        List of transforms to apply to target image.

    """

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        path_csv: Optional[str] = None,
        transform_signal: Optional[list] = None,
        transform_target: Optional[list] = None,
    ):
        self.path_csv = None
        if dataframe is not None:
            self.df = dataframe
        else:
            self.path_csv = path_csv
            self.df = pd.read_csv(self.path_csv)
        self.transform_signal = to_objects(transform_signal)
        self.transform_target = to_objects(transform_target)
        self._metadata = None
        self.loc = _LocIndexer(self)
        self.iloc = _iLocIndexer(self)

    @property
    def metadata(self) -> dict:
        """Returns metadata about the dataset."""
        if self._metadata is not None:
            return self._metadata
        self._metadata = {}
        if self.path_csv is not None:
            self._metadata["path_csv"] = self.path_csv
        self._metadata["transform_signal"] = _to_str_list(self.transform_signal)
        self._metadata["transform_target"] = _to_str_list(self.transform_target)
        return self._metadata

    def get_information(self, index) -> Union[dict, str]:
        """Returns information to identify dataset element specified by index.

        """
        raise NotImplementedError
