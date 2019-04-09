import os

import pandas as pd

from fnet.data.tiffdataset import TiffDataset
from fnet.utils.general_utils import add_augmentations
import fnet


def testdataset(train: bool = False) -> TiffDataset:
    """Dummy dataset for testing."""
    path_data_dir = os.path.join(
        os.path.dirname(fnet.__file__), os.pardir, 'data'
    )
    df = pd.DataFrame({
        'path_signal': [os.path.join(path_data_dir, 'EM_low.tif')],
        'path_target': [os.path.join(path_data_dir, 'MBP_low.tif')],
    }).rename_axis('arbitrary')
    if not train:
        df = add_augmentations(df)
    return TiffDataset(dataframe=df)
