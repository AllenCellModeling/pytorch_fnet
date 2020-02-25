from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import tifffile
from aicsimageio.writers import OmeTiffWriter


def create_data_dir(path_root: Path):
    path_data = path_root / "data"
    path_data.mkdir(exist_ok=True)

    return path_data


def create_tif_data(
    path_root: Path, shape: Sequence[int], n_items: int, weights: bool
) -> Path:
    path_data = create_data_dir(path_root)

    records = []

    for idx in range(n_items):
        path_x = path_data / f"{idx:02}_x.tif"
        path_y = path_data / f"{idx:02}_y.tif"
        data_x = np.random.randint(128, size=shape, dtype=np.uint8)
        data_y = data_x + idx

        tifffile.imsave(path_x, data_x, compress=2)
        tifffile.imsave(path_y, data_y, compress=2)

        records.append({"dummy_id": idx, "path_signal": path_x, "path_target": path_y})
        if weights:
            # Create map that covers half of each dim
            data_weight_map = np.zeros(shape, dtype=np.float32)
            slicey = [slice(shape[d] // 2) for d in range(len(shape))]
            slicey = tuple(slicey)
            data_weight_map[slicey] = 1
            path_weight_map = path_data / f"{idx:02}_weight_map.tif"
            tifffile.imsave(path_weight_map, data_weight_map, compress=2)
            records[-1]["path_weight_map"] = path_weight_map

    path_csv = path_root / "dummy.csv"
    pd.DataFrame(records).set_index("dummy_id").to_csv(path_csv)
    return path_csv


def create_multichtiff_data(
    path_root: Path, dims_zyx: Sequence[int], n_ch_in: int, n_ch_out: int, n_items: int
) -> Path:

    assert len(dims_zyx) == 3

    path_data = create_data_dir(path_root)

    records = []

    for idx in range(n_items):
        path_x = path_data / f"{idx:02}.tif"
        data_x = np.random.randint(
            128, size=[n_ch_in + n_ch_out] + list(dims_zyx), dtype=np.uint8
        )

        with OmeTiffWriter(path_x) as writer:
            writer.save(data_x, dimension_order="CZYX")  # should be a numpy array

        records.append(
            {
                "dummy_id": idx,
                "path_tiff": path_x,
                "channel_signal": list(np.arange(0, n_ch_in)),
                "channel_target": list(np.arange(0, n_ch_out) + n_ch_in),
            }
        )

    path_csv = path_root / "dummy.csv"
    pd.DataFrame(records).set_index("dummy_id").to_csv(path_csv)

    return path_csv
