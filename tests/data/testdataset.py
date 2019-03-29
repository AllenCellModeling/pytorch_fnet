from fnet.data.tiffdataset import TiffDataset
from fnet.utils.general_utils import add_augmentations
import fnet
import pandas as pd
import os
import pdb


def testdataset(train: bool) -> TiffDataset:
    """Dummy dataset for testing."""
    path_data_dir = os.path.join(
        os.path.dirname(fnet.__file__), os.pardir, 'data'
    )
    df = pd.DataFrame({
        'path_signal': [os.path.join(path_data_dir, 'EM_low.tif')],
        'path_target': [os.path.join(path_data_dir, 'MBP_low.tif')],
    })
    if not train:
        df = add_augmentations(df)
        df = df.iloc[1:, :].reset_index(drop=True)
    return TiffDataset(dataframe=df)
