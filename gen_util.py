import aicsimage.processing as proc
import aicsimage.io as io
import numpy as np

class CziLoader(object):
    def __init__(self, file_path):
        # Currently, expect to deal only with CZI files where 'B' and '0' dimensions are size 1
        self.czi_reader = io.cziReader.CziReader(file_path)
        czi_np = self.czi_reader.czi.asarray()
        assert (czi_np.shape[0], czi_np.shape[-1]) == (1, 1), \
            "'B' and '0' dimensions are not size 1"
        self.czi_np = czi_np
        
    def get_volume(self, c):
        """Returns the image volume for the specified channel."""
        if self.czi_reader.hasTimeDimension:
            raise NotImplementedError  # TODO: handle case of CZI images with T dimension
        if self.czi_reader.czi.axes == b'BCZYX0':
            return self.czi_np[0, c, :, :, :, 0]

def extract_chunk(larger_ar, chunk_dims, coord):
    """Returns smaller array extracted from a larger array.

    Parameters:
    larger_ar - numpy.array
    chunk_dims - tuple of chunk dimensions
    coord - tuple to indicate coordinate in larger_ar to start the extraction. If None,
      a random valid coordinate will be selected
    """
    if coord is None:
        assert len(larger_ar.shape) == len(chunk_dims)
        coord = [0, 0, 0]
        for i in range(len(chunk_dims)):
            coord[i] = np.random.random_integers(0, larger_ar.shape[i] - chunk_dims[i])
    else:
        assert len(larger_ar.shape) == len(chunk_dims) == len(coord)
    slices = []
    for i in range(len(coord)):
        slices.append(slice(coord[i], coord[i] + chunk_dims[i]))
    return larger_ar[slices]

        
def main():
    fname = './test_images/20161209_C01_001.czi'
    print('main')
    czi_loader = CziLoader(fname)
    vol = czi_loader.get_volume(0)
    print(type(vol))
    print(vol.shape)
    test = extract_chunk(vol, (2,3,4), (0, 0, 0))
    print(test.shape)
    print(test)

    # chunk same size as original
    test = extract_chunk(vol, vol.shape, None)
    print(test.shape)

    for i in range(10):
        test = extract_chunk(vol, (2,3,4), None)

if __name__ == '__main__':
    main()
