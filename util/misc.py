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
    return z_max

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
    print('min:', ar.min(), '| max:', ar.max(), '| median', np.median(ar))

def save_img_np(img_np, path):
    """Save image (numpy array, ZYX) as a TIFF."""
    with omeTifWriter.OmeTifWriter(path, overwrite_file=True) as fo:
        fo.save(img_np)
        print('saved:', path)
            
