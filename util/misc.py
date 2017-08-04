import os
import numpy as np
from aicsimage.io import omeTifWriter
import pdb

__all__ = [
    'find_z_of_max_slice',
    'find_z_max_std',
    'get_vol_transformed',
    'pad_mirror',
    'print_array_stats',
    'save_img_np'
]

def find_z_of_max_slice(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice the highest total pixel intensity."""
    z_max = np.argmax(np.sum(ar, axis=(1, 2)))
    return int(z_max)

def find_z_max_std(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice with the highest pixel intensity std."""
    z_max = np.argmax(np.std(ar, axis=(1, 2)))
    return int(z_max)

def get_vol_transformed(ar, transform):
    """Apply the transformation(s) to the supplied array and return the result."""
    result = ar
    if transform is None:
        pass
    elif isinstance(transform, (list, tuple)):
        for t in transform:
            result = t(result)
    else:
        result = transform(result)
    return result

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

def print_array_stats(ar):
    print('shape:', ar.shape, '|', 'dtype:', ar.dtype)
    stat_list = []
    stat_list.append('min: {:.3f}'.format(ar.min()))
    stat_list.append('max: {:.3f}'.format(ar.max()))
    stat_list.append('median: {:.3f}'.format(np.median(ar)))
    stat_list.append('mean: {:.3f}'.format(np.mean(ar)))
    stat_list.append('std: {:.3f}'.format(np.std(ar)))
    print(' | '.join(stat_list))
    # print('min:', ar.min(), '| max:', ar.max(), '| median', np.median(ar))

def save_img_np(img_np, path):
    """Save image (numpy array, ZYX) as a TIFF."""
    path_dirname = os.path.dirname(path)
    if not os.path.exists(path_dirname):
        os.makedirs(path_dirname)
    with omeTifWriter.OmeTifWriter(path, overwrite_file=True) as fo:
        fo.save(img_np)
        print('saved tif:', path)
            
