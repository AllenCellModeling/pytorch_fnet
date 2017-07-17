import numpy as np
import aicsimage.processing as proc

def sub_mean_norm(img):
    """Subtract mean, set STD to 1.0"""
    result = img.copy()
    result -= np.mean(img)
    result /= np.std(img)
    return result

def do_nothing(img):
    pass

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
