from typing import Tuple

import numpy as np
import pytest

from fnet.data import BufferedPatchDataset

# Make sure each item is used
# Test 1d, 2d, 3d


class _DummyDataset:

    def __init__(self, nd: int = 1):
        self.data = []
        for idx in range(8):
            x = np.ones((8,)*nd, dtype=np.int)*idx
            y = x**2
            self.data.append((x, y))
        self.accessed = set()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        self.accessed.add(index)
        return self.data[index]


def test_bufferedpatchdataset_bad_input():
    ds = _DummyDataset()

    # Too many patch_size dimensions
    bpds = BufferedPatchDataset(ds)
    with pytest.raises(ValueError):
        next(bpds)

    # patch_size too big
    bpds = BufferedPatchDataset(ds, patch_size=(9,))
    with pytest.raises(ValueError):
        next(bpds)

    # Inconsistant spatial shape
    bad = [part for part in ds.data[0]]
    bad[0] = bad[0][1:]
    ds.data[0] = tuple(bad)
    bpds = BufferedPatchDataset(ds, patch_size=(4,), shuffle_images=False)
    with pytest.raises(ValueError):
        next(bpds)


if __name__ == '__main__':
    test_bufferedpatchdataset_bad_input()
