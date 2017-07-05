import numpy as np

__all__ = ['find_z_of_max_slice', 'print_array_stats']

def find_z_of_max_slice(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice with the most signal."""
    z_max = np.argmax(np.sum(ar, axis=(1, 2)))
    return z_max

def print_array_stats(ar):
    print('shape:', ar.shape, '|', 'dtype:', ar.dtype)
    print('min:', ar.min(), '| max:', ar.max(), '| median', np.median(ar))
