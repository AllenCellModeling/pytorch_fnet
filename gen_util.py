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

def pick_random_chunk_coord(shape, dims_chunk, dims_pin=(None, None, None)):
    """Returns a random coordinate from where an array chunk can be extracted from a larger array.
    
    Parameters:
    shape - tuple indicating shape of larger array
    dims_chunk - tuple of chunk dimensions
    dims_pin (optional) - tuple of fixed coordinate values. Dimensions for the returned coordinate
      will be set to the dims_pin value if the value is not None.
    
    Returns:
    coord - tuple of coordinate within larger array
    """
    assert len(shape) == len(dims_chunk)
    coord = [0, 0, 0]
    for i in range(len(dims_chunk)):
        if dims_pin[i] is None:
            coord[i] = np.random.random_integers(0, shape[i] - dims_chunk[i])
        else:
            coord[i] = dims_pin[i]
    return tuple(coord)

def extract_chunk(larger_ar, dims_chunk, coord):
    """Returns smaller array extracted from a larger array.

    Parameters:
    larger_ar - numpy.array
    dims_chunk - tuple of chunk dimensions
    coord - tuple to indicate coordinate in larger_ar to start the extraction. If None,
      a random valid coordinate will be selected
    """
    assert len(larger_ar.shape) == len(dims_chunk) == len(coord)
    slices = []
    for i in range(len(coord)):
        slices.append(slice(coord[i], coord[i] + dims_chunk[i]))
    return larger_ar[slices]

        
def main():
    fname = './test_images/20161209_C01_001.czi'
    print('main')
    czi_loader = CziLoader(fname)
    vol = czi_loader.get_volume(0)
    print('larger array shape:', vol.shape)
    chunk = extract_chunk(vol, (2,3,4), (0, 0, 0))
    print('chunk shape:', chunk.shape)
    print(chunk)

    # random coordinate selection
    dims_chunk = (5,6,7)

    n_runs = 10000
    coords_random = np.zeros((n_runs, len(dims_chunk)), dtype=np.int)
    print('generating', n_runs, 'random coordinates for chunk of shape', dims_chunk)
    dims_pin = (None, None, None)
    for i in range(n_runs):
        coord = pick_random_chunk_coord(vol.shape, dims_chunk, dims_pin)
        # print('random coordinate for chunk of size', dims_chunk, '->', coord)
        coords_random[i] = coord
        # test = extract_chunk(vol, dims_chunk, coord)
        # print('extracted chunk shape:', test.shape)
    print('random coord mins:', np.amin(coords_random, axis=0))
    print('random coord maxs:', np.amax(coords_random, axis=0))
    print('first 5 random coords:')
    print(coords_random[:5])

def test_czireader():
    fname = './test_images/20161209_C01_001.czi'
    reader = io.cziReader.CziReader(fname)
    print('axes:', reader.czi.axes)
    czi_np = reader.load()
    print(type(czi_np), czi_np.shape)

if __name__ == '__main__':
    main()
    # test_czireader()
