from typing import Sequence

import numpy as np
import pytest

from fnet.data import multichtiffdataset
from .data.testlib import create_multichtiff_data


@pytest.mark.parametrize(
    "n_ch_in, n_ch_out, dims_zyx",
    [(1, 1, (64, 128, 32)), (3, 1, (12, 13, 14)), (5, 5, (12, 13, 14))],
)
def test_MultiTiffDataset(tmp_path, n_ch_in, n_ch_out, dims_zyx):
    """Tests TiffDataset class."""
    n_items = 5
    path_dummy = create_multichtiff_data(
        tmp_path, n_ch_in=n_ch_in, n_ch_out=n_ch_out, dims_zyx=dims_zyx, n_items=n_items
    )
    ds = multichtiffdataset.MultiChTiffDataset(path_csv=path_dummy)

    assert len(ds) == n_items
    idx = n_items // 2
    info = ds.get_information(n_items // 2)
    assert isinstance(info, dict)
    assert all(col in info for col in ds.df.columns)
    data = ds[idx]
    len_data = 2
    assert len(data) == len_data

    assert tuple(data[0].shape) == (n_ch_in,) + dims_zyx
    assert tuple(data[1].shape) == (n_ch_out,) + dims_zyx
