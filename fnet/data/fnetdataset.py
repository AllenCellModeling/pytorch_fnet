from fnet.utils.general_utils import to_objects
from typing import Optional, Union
import pandas as pd
import pdb  # noqa: F401
import torch.utils.data


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
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
        self.transform_signal = to_objects(transform_signal)
        self.transform_target = to_objects(transform_target)

    def get_information(self, index) -> Union[dict, str]:
        """Returns information to identify dataset element specified by index.

        """
        raise NotImplementedError
