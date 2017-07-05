import numpy as np

def normalize(img):
    """Adjust pixel intensity to (0.0, 1.0)"""
    img -= np.amin(img)
    img /= np.amax(img)

def do_nothing(img):
    pass
