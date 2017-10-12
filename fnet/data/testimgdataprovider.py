import pdb
import numpy as np
from fnet import get_vol_transformed

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

    def using_train_set(self):
        return self._dataset._train_select

    def use_test_set(self):
        self._dataset.use_test_set()
        
    def use_train_set(self):
        self._dataset.use_train_set()

    def get_name(self, i):
        """Returns a name representing element i."""
        return self._dataset.get_name(i, 0)
    
    def _make_batch(self, volumes_tuple):
        """Change supplied 3d arrays into 5d batch array."""
        data = []
        for vol in volumes_tuple:
            shape = (1, 1, ) + vol.shape
            data.append(np.zeros(shape, dtype=vol.dtype))
            data[-1][0, 0, ] = vol
        return tuple(data)

    def __repr__(self):
        return 'DataProvider({:d} elements)'.format(len(self))
        
    def __len__(self):
        return len(self._dataset)
    
    def get_item_sel(self, idx, sel):
        volume_pre = self._dataset.get_item_sel(idx, sel)
        if self._transforms is None:
            volume = volume_pre
        else:
            volume = get_vol_transformed(volume_pre, self._transforms[sel])
        data_tuple = self._make_batch((volume,))
        return data_tuple[0]

    def __getitem__(self, idx):
        volumes_pre = self._dataset[idx]
        if volumes_pre is None:
            return None
        volumes = []
        for i, volume_pre in enumerate(volumes_pre):
            if self._transforms is None:
                if volume_pre is None:
                    return None
                volumes.append(volume_pre)
            else:
                vol_trans = get_vol_transformed(volume_pre, self._transforms[i])
                if vol_trans is None:
                    return None
                volumes.append(vol_trans)
        data_tuple = self._make_batch(volumes)
        return data_tuple
