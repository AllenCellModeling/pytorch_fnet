import numpy as np
import aicsimage.processing as proc

def normalize(img):
    """Adjust pixel intensity to (0.0, 1.0)"""
    img -= np.amin(img)
    img /= np.amax(img)

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
