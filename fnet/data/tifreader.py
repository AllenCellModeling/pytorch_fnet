import pdb  # noqa: F401
import tifffile


class TifReader(object):
    def __init__(self, filepath):
        self.tif_np = tifffile.imread(filepath)

    def get_image(self):
        """Returns the image as NumPy array."""
        return self.tif_np
