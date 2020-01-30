from typing import Sequence

import numpy as np
import pytest

from fnet.data import tiffdataset
from .data.testlib import create_tif_data


@pytest.mark.parametrize(
    "shape,weights", [((16, 32), False), ((8, 16, 32), False), ((8, 16, 32), True)]
)
def test_TiffDataset(tmp_path, shape: Sequence[int], weights: bool):
    """Tests TiffDataset class."""
    n_items = 5
    path_dummy = create_tif_data(
        tmp_path, shape=shape, n_items=n_items, weights=weights
    )
    ds = tiffdataset.TiffDataset(path_csv=path_dummy, col_index="dummy_id")
    assert len(ds) == n_items
    idx = n_items // 2
    info = ds.get_information(n_items // 2)
    assert isinstance(info, dict)
    assert all(col in info for col in ds.df.columns)
    data = ds[idx]
    len_data = 3 if weights else 2
    assert len(data) == len_data
    shape_exp = (1,) + shape
    for d in data:
        assert tuple(d.shape) == shape_exp

    factor = int((data[1] - data[0]).numpy().mean())
    assert factor == idx

    if weights:
        weight_sum_exp = np.prod([d // 2 for d in shape])
        weight_sum_got = int(data[-1].numpy().sum())
        assert weight_sum_got == weight_sum_exp
