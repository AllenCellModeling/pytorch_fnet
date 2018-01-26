import numpy as np
import scipy
import pdb
import warnings
import pdb

def normalize(img):
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
        self.crops = {}

    def _adjust_shape_crop(self, shape_crop):
        key = tuple(shape_crop)
        if key in self._shape_adjustments:
            return self._shape_adjustments[key]
        shape_crop_new = list(shape_crop)
        prod_shape = np.prod(shape_crop_new)
        idx_dim_reduce = 0

        if len(shape_crop) == 3:
            order_dim_reduce = [2, 1, 2, 2, 1, 0]
        else:
            order_dim_reduce = [0, 1]

        while prod_shape > self.n_max_pixels:
            dim = order_dim_reduce[idx_dim_reduce]
            if not (dim == 0 and shape_crop_new[dim] <= 64):
                print('DEBUG: reducing dim', dim)
                shape_crop_new[dim] -= self.reduce_by
                prod_shape = np.prod(shape_crop_new)
            idx_dim_reduce += 1
            if idx_dim_reduce >= len(order_dim_reduce):
                idx_dim_reduce = 0
        value = tuple(shape_crop_new)
        self._shape_adjustments[key] = value
        print('DEBUG: cropper shape change', shape_crop, 'becomes', value)
        return value

    def _get_shape_crop(self, x):
        shape_crop = []
        for i in range(len(self.shape)):
            if self.shape[i] is None:
                raise NotImplementedError
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
        return shape_crop

    def __call__(self, x):
        shape_input = x.shape
        if shape_input in self.crops:
            slices = self.crops[shape_input]['slices']
            print('DEBUG: using stored calculation: {} -> {}'.format(shape_input, slices))
            return x[slices].copy()
        shape_crop = []
        for i in range(len(self.shape)):
            if self.shape[i] is None:
                shape_crop.append(shape_input[i])
            elif isinstance(self.shape[i], int):
                if self.shape[i] > x.shape[i]:
                    warnings.warn('Crop dimensions larger than image dimension ({} > {} for dim {}).'.format(self.shape[i], x.shape[i], i))
                    raise AttributeError
                shape_crop.append(self.shape[i])
            elif isinstance(self.shape[i], str):  # e.g., '/16'
                multiple_of = int(self.shape[i][1:])
                shape_crop.append(x.shape[i] & ~(multiple_of - 1))
            else:
                raise NotImplementedError
        shape_crop = self._adjust_shape_crop(shape_crop)
        slices = []
        offsets_crop = []
        for i in range(len(self.shape)):
            if self.offsets[i] == 'mid':  # take crop from middle of input array dim
                offset = (x.shape[i] - shape_crop[i])//2
            else:
                offset = self.offsets[i]
            if offset + shape_crop[i] > x.shape[i]:
                warnings.warn('Cannot crop outsize image dimensions ({}:{} for dim {}). Starting crop from 0 instead.'.format(offset, offset + shape_crop[i], i))
                offset = 0
            slices.append(slice(offset, offset + shape_crop[i]))
            offsets_crop.append(offset)
        # print('DEBUG: shape', x[slices].shape, '| pixels', x[slices].size)
        self.crops[shape_input] = {
            'shape_input': shape_input,
            'shape_crop': shape_crop,
            'offsets_crop': offsets_crop,
            'slices': slices,
        }
        self.crops['last'] = self.crops[shape_input]
        return x[slices].copy()

    def unprop(self, ar):
        print(self.crops['last'])
        shape_input = self.crops['last']['shape_input']
        slices = self.crops['last']['slices']
        ar_unpropped = np.zeros(shape_input, dtype=ar.dtype)
        ar_unpropped[slices] = ar
        return ar_unpropped

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

    
def pad_mirror(ar, padding):
    """Pad 3d array using mirroring.

    Parameters:
    ar - (numpy.array) array to be padded
    padding - (tuple) per-dimension padding values
    """
    shape = tuple((ar.shape[i] + 2*padding[i]) for i in range(3))
    result = np.zeros(shape, dtype=ar.dtype)
    slices_center = tuple(slice(padding[i], padding[i] + ar.shape[i]) for i in range(3))
    result[slices_center] = ar
    # z-axis, centers
    if padding[0] > 0:
        result[0:padding[0], slices_center[1] , slices_center[2]] = np.flip(ar[0:padding[0], :, :], axis=0)
        result[ar.shape[0] + padding[0]:, slices_center[1] , slices_center[2]] = np.flip(ar[-padding[0]:, :, :], axis=0)
    # y-axis
    result[:, 0:padding[1], :] = np.flip(result[:, padding[1]:2*padding[1], :], axis=1)
    result[:, padding[1] + ar.shape[1]:, :] = np.flip(result[:, ar.shape[1]:ar.shape[1] + padding[1], :], axis=1)
    # x-axis
    result[:, :, 0:padding[2]] = np.flip(result[:, :, padding[2]:2*padding[2]], axis=2)
    result[:, :, padding[2] + ar.shape[2]:] = np.flip(result[:, :, ar.shape[2]:ar.shape[2] + padding[2]], axis=2)
    return result
