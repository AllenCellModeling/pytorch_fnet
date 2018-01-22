import aicsimage.io as io
import os
import pdb



class TifReader(object):
    def __init__(self, filepath):
        with io.tifReader.TifReader(filepath) as reader:
            """Keeping it this way in order to extend it further for multi-channel tifs"""
            self.tif_np = reader.tif.asarray()
         

    def get_image(self):
        """Returns the image for the specified channel."""
        """Keeping it this way in order to extend it further for multi-channel tifs"""

        return self.tif_np
