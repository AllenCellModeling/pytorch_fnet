from collections import Counter
from typing import Tuple

import numpy as np
import numpy.testing as npt
import pytest

from fnet.data import BufferedPatchDataset


class _DummyDataset:

    def __init__(self, nd: int = 1):
        self.data = []
        shape = (8,)*nd
        for idx in range(8):
            x = np.arange(idx, idx + 8**nd).reshape(shape)
            y = x**2
            self.data.append((x, y))
        self.accessed = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        self.accessed.append(index)
        return self.data[index]


def test_bad_input():
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


@pytest.mark.parametrize('nd', [2, 3])
def test_nd(nd: int):
    """Checks shape of returned item and checks that all dataset elements were
    accessed.

    Parameters
    ----------
    nd
        Number of spatial dimensions for dataset elements.

    """
    ds = _DummyDataset(nd)
    patch_size = tuple(range(2, nd + 2))
    interval = 3
    buffer_size = 2
    bpds = BufferedPatchDataset(
        ds,
        patch_size=patch_size,
        buffer_size=buffer_size,
        buffer_switch_frequency=interval,
    )
    # Sample enough patches such that the entire dataset is used twice
    n_swaps = 2*len(ds) - buffer_size
    for _idx in range(n_swaps*interval):
        x, y = next(bpds)
        assert x.shape == patch_size
        npt.assert_array_equal(y, x**2)
        assert bpds.get_buffer_history() == ds.accessed
    counts = Counter(ds.accessed)
    assert max(counts.values()) == min(counts.values()) == 2


def test_sampling():
    """Verifies that samples are pulled from entire range of dataset items."""
    ds = _DummyDataset(nd=3)
    x_low, x_hi = float('inf'), float('-inf')
    y_low, y_hi = float('inf'), float('-inf')
    bpds = BufferedPatchDataset(
        ds,
        patch_size=(7, 7, 7),
        buffer_size=1,
        buffer_switch_frequency=-1,
        shuffle_images=False,
    )
    # Patch locations are randomized, so look at many patches and check that
    # the ends of the dataset item are sampled at least once.
    for _idx in range(128):
        x, y = next(bpds)
        x_low = min(x_low, x.min())
        x_hi = max(x_hi, x.max())
        y_low = min(y_low, y.min())
        y_hi = max(y_hi, y.max())
    assert x_low == y_low == 0
    assert x_hi == 511
    assert x_hi**2 == y_hi


def test_smaller_patch():
    """Verifies that patches smaller than the dataset item are pulled from the
    last dataset item dimensions.

    """
    nd = 4
    ds = _DummyDataset(nd=nd)
    patch_size = (4,)*(nd - 1)
    bpds = BufferedPatchDataset(
        ds,
        patch_size=patch_size,
        buffer_size=1,
        buffer_switch_frequency=-1,
        shuffle_images=False,
    )
    x, y = next(bpds)
    assert x.shape == y.shape == ((8,) + patch_size)
