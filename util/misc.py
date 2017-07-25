import os
import numpy as np
from aicsimage.io import omeTifWriter

__all__ = [
    'find_z_of_max_slice',
    'get_vol_transformed',
    'print_array_stats',
    'save_img_np'
]

def find_z_of_max_slice(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice with the most signal."""
    z_max = np.argmax(np.sum(ar, axis=(1, 2)))
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
            
