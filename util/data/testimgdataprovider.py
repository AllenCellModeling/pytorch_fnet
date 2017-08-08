import pdb
import numpy as np
from util.misc import pad_mirror
from util import get_vol_transformed

# For each element in a dataset, provides 1 "batch" of size 1.
class TestImgDataProvider(object):
    def __init__(self, dataset,
                 transforms=None):
        """
        Parameter:
        dataset - DataSet instance
        transforms - list/tuple of transforms to apply to dataset element
        """
        self._dataset = dataset
        self._transforms = transforms
        
        self._count_iter = 0
        self._n_iter = len(dataset)

    def _make_batch(self, volumes_tuple):
        """Change supplied 3d arrays into 5d batch array."""
        data = []
        for vol in volumes_tuple:
            shape = (1, 1, ) + vol.shape
            data.append(np.zeros(shape, dtype=vol.dtype))
            data[-1][0, 0, ] = vol
        return tuple(data)
        
    def __len__(self):
        return len(dataset)

    def __iter__(self):
        self._count_iter = 0
        return self

    def __next__(self):
        if self._count_iter >= self._n_iter:
            raise StopIteration
        volumes_pre = self._dataset[self._count_iter]
        volumes = []
        for i, volume_pre in enumerate(volumes_pre):
            if self._transforms is None:
                volumes.append(volume_pre)
            else:
                volumes.append(get_vol_transformed(volume_pre, self._transforms[i]))
            print('DEBUG: shape change', volume_pre.shape, volumes[-1].shape)
        data_tuple = self._make_batch(volumes)
        self._count_iter += 1
        return data_tuple
