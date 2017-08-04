import numpy as np
import aicsimage.processing as proc
from util.misc import pad_mirror
import warnings

def sub_mean_norm(img):
    """Subtract mean, set STD to 1.0"""
    result = img.copy()
    result -= np.mean(img)
    result /= np.std(img)
    return result

def do_nothing(img):
    return img

class Cropper(object):
    def __init__(self, shape, offsets=None):
        """Crop input array to given shape."""
        assert isinstance(shape, (list, tuple))
        if offsets:
            assert len(offsets) == len(shape)

        self._shape = tuple(shape)
        self._offsets = tuple(offsets) if offsets is not None else (0, )*len(shape)
    
    def __call__(self, x):
        slices = []
        for i in range(len(self._shape)):
            if self._shape[i] is None:
                start = 0
                end = x.shape[i]
            else:
                start = self._offsets[i]
                if start + self._shape[i] > x.shape[i]:
                    warnings.warn('Cannot crop outsize image dimensions ({}:{} for dim {}). Starting crop from 0 instead.'.format(start, start + self._shape[i], i))
                    start = 0
                end = (start + self._shape[i])
            slices.append(slice(start, end))
        print('DEBUG: cropper', slices)
        return x[slices].copy()

    def __str__(self):
        params = [str(self._shape)]
        if self._offsets:
            params.append(str(self._offsets))
        str_out = 'Cropper{}'.format(', '.join(params))
        return str_out

    

class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self._factors = factors

    def __call__(self, x):
        return proc.resize(x, self._factors)

    def __str__(self):
        str_out = 'Resizer | factors: ' + str(self._factors) 
        return str_out

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
