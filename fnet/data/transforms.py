import numpy as np
import scipy
from fnet import pad_mirror
import warnings
import pdb

def sub_mean_norm(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

def do_nothing(img):
    return img

class Cropper(object):
    def __init__(self, shape, offsets=None, n_max_pixels=9732096, reduce_by=16):
        """Crop input array to given shape."""
        assert isinstance(shape, (list, tuple))
        for i in shape:
            if isinstance(i, int):
                assert i >= 0
            elif isinstance(i, str):
                assert int(i[1:]) in [4, 8, 16, 32, 64]
        if offsets:
            assert len(offsets) == len(shape)

        self.shape = tuple(shape)
        self.offsets = tuple(offsets) if offsets is not None else (0, )*len(shape)
        self.n_max_pixels = n_max_pixels
        self.reduce_by = reduce_by
        self._shape_adjustments = {}

    def _adjust_shape_crop(self, shape_crop):
        key = tuple(shape_crop)
        if key in self._shape_adjustments:
            return self._shape_adjustments[key]
        shape_crop_new = list(shape_crop)
        prod_shape = np.prod(shape_crop_new)
        idx_dim_reduce = 0
        if shape_crop[0] <= 64:
            order_dim_reduce = [2, 1, 2, 1, 2, 1, 0]
        else:
            order_dim_reduce = [2, 1, 2, 1, 0]
        while prod_shape > self.n_max_pixels:
            dim = order_dim_reduce[idx_dim_reduce]
            shape_crop_new[dim] -= self.reduce_by
            prod_shape = np.prod(shape_crop_new)
            idx_dim_reduce += 1
            if idx_dim_reduce >= len(order_dim_reduce):
                idx_dim_reduce = 0
        value = tuple(shape_crop_new)
        self._shape_adjustments[key] = value
        return value
    
    def __call__(self, x):
        shape_crop = []
        for i in range(len(self.shape)):
            if self.shape[i] is None:
                shape_crop.append(len_dim)
            elif isinstance(self.shape[i], int):
                if self.shape[i] > x.shape[i]:
                    warnings.warn('Crop dimensions larger than image dimension ({} > {} for dim {}).'.format(self.shape[i], x.shape[i], i))
                    return None
                shape_crop.append(self.shape[i])
            elif isinstance(self.shape[i], str):  # e.g., '/16'
                multiple_of = int(self.shape[i][1:])
                shape_crop.append(x.shape[i] & ~(multiple_of - 1))
            else:
                raise NotImplementedError
        shape_crop = self._adjust_shape_crop(shape_crop)
        slices = []
        for i in range(len(self.shape)):
            if self.offsets[i] == 'mid':  # take crop from middle of input array dim
                offset = (x.shape[i] - shape_crop[i])//2
            else:
                offset = self.offsets[i]
            if offset + shape_crop[i] > x.shape[i]:
                warnings.warn('Cannot crop outsize image dimensions ({}:{} for dim {}). Starting crop from 0 instead.'.format(offset, offset + shape_crop[i], i))
                offset = 0
            slices.append(slice(offset, offset + shape_crop[i]))
        # print('DEBUG: shape', x[slices].shape, '| pixels', x[slices].size)
        return x[slices].copy()

    def __str__(self):
        params = [str(self.shape)]
        if self._offsets:
            params.append(str(self._offsets))
        str_out = 'Cropper{}'.format(', '.join(params))
        return str_out

class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self.factors = factors

    def __call__(self, x):
        return scipy.ndimage.zoom(x, (self.factors), mode='nearest')

    def __repr__(self):
        return 'Resizer({:s})'.format(str(self.factors)) 

class ReflectionPadder3d(object):
    def __init__(self, padding):
        """Return padded 3D numpy array by mirroring/reflection.

        Parameters:
        padding - (int or tuple) size of the padding. If padding is an int, pad all dimensions by the same value. If
        padding is a tuple, pad the (z, y, z) dimensions by values specified in the tuple."""
        self._padding = None
        
        if isinstance(padding, int):
            self._padding = (padding, )*3
        elif isinstance(padding, tuple):
            self._padding = padding
        if (self._padding == None) or any(i < 0 for i in self._padding):
            raise AttributeError

    def __call__(self, ar):
        return pad_mirror(ar, self._padding)

class Capper(object):
    def __init__(self, low=None, hi=None):
        self._low = low
        self._hi = hi
        
    def __call__(self, ar):
        result = ar.copy()
        if self._hi is not None:
            result[result > self._hi] = self._hi
        if self._low is not None:
            result[result < self._low] = self._low
        return result

    def __repr__(self):
        return 'Capper({}, {})'.format(self._low, self._hi)
