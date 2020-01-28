from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tifffile


def create_tif_data(
    path_root: Path, shape: Sequence[int], n_items: int, weights: bool
) -> Path:
    """Creates dummy tif data."""
    path_data = path_root / "data"
    path_data.mkdir(exist_ok=True)
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
