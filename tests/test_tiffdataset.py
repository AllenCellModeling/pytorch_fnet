from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import tifffile

from fnet.data import tiffdataset


def _create_data(
        path_root: Path, shape: Sequence[int], n_items: int, weights: bool
) -> Path:
    """Creates dummy data."""
    path_data = path_root / 'data'
    path_data.mkdir(exist_ok=True)
    records = []
    for idx in range(n_items):
        path_x = path_data / f'{idx:02}_x.tif'
        path_y = path_data / f'{idx:02}_y.tif'
        data_x = np.random.randint(128, size=shape, dtype=np.uint8)
        data_y = data_x + idx
        tifffile.imsave(path_x, data_x, compress=2)
        tifffile.imsave(path_y, data_y, compress=2)
        records.append(
            {
                'dummy_id': idx,
                'path_signal': path_x,
                'path_target': path_y,
            }
        )
        if weights:
            # Create map that covers half of each dim
            data_weight_map = np.zeros(shape, dtype=np.float32)
            slicey = [slice(shape[d]//2) for d in range(len(shape))]
            slicey = tuple(slicey)
            data_weight_map[slicey] = 1
            path_weight_map = path_data / f'{idx:02}_weight_map.tif'
            tifffile.imsave(path_weight_map, data_weight_map, compress=2)
            records[-1]['path_weight_map'] = path_weight_map
    path_csv = path_root / 'dummy.csv'
    pd.DataFrame(records).set_index('dummy_id').to_csv(path_csv)
    return path_csv


@pytest.mark.parametrize(
    'shape,weights',
    [((16, 32), False), ((8, 16, 32), False), ((8, 16, 32), True)],
)
def test_TiffDataset(tmp_path, shape: Sequence[int], weights: bool):
    """Tests TiffDataset class."""
    n_items = 5
    path_dummy = _create_data(
        tmp_path, shape=shape, n_items=n_items, weights=weights
    )
    ds = tiffdataset.TiffDataset(path_csv=path_dummy, col_index='dummy_id')
    assert len(ds) == n_items
    idx = n_items // 2
    info = ds.get_information(n_items // 2)
    assert isinstance(info, dict)
    assert all(col in info for col in ds.df.columns)
    data = ds[idx]
    len_data = 3 if weights else 2
    assert len(data) == len_data
    shape_exp = (1, ) + shape
    for d in data:
        assert tuple(d.shape) == shape_exp
    factor = int((data[1] - data[0]).numpy().mean())
    assert factor == idx
    if weights:
        weight_sum_exp = np.prod([d // 2 for d in shape])
        weight_sum_got = int(data[-1].numpy().sum())
        assert weight_sum_got == weight_sum_exp
