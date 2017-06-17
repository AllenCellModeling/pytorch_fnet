import aicsimage.processing as proc
import aicsimage.io as io
import numpy as np
import models
import pdb

class Loader(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.signal_np = None
        self.target_np = None

    def get_batch(self, n, dims_chunk=(32, 32, 32), dims_pin=(None, None, None)):
        """Get a batch of examples from source data."

        Parameters:
        n - (int) batch size.
        dims_chunk - (tuple) ZYX dimensions of each example.
        
        Returns:
        batch_x, batch_y - (2 numpy arrays) each array will have shape (n, 1) + dims_chunk.
        """
        shape_batch = (n, 1) + dims_chunk
        batch_x = np.zeros(shape_batch)
        batch_y = np.zeros(shape_batch)
        coords = self._pick_random_chunk_coord(dims_chunk, n=n, dims_pin=dims_pin)
        for i in range(len(coords)):
            coord = coords[i]
            # print(coord)
            chunks_tup = self._extract_chunk(dims_chunk, coord)
            batch_x[i, 0, ...] = chunks_tup[0]
            batch_y[i, 0, ...] = chunks_tup[1]
        return batch_x, batch_y

    def _pick_random_chunk_coord(self, dims_chunk, n=1, dims_pin=(None, None, None)):
        """Returns a random coordinate from where an array chunk can be extracted from signal_, target_np.

        Parameters:
        dims_chunk - tuple of chunk dimensions
        n - (int, optional) - 
        dims_pin (optional) - tuple of fixed coordinate values. Dimensions for the returned coordinate
          will be set to the dims_pin value if the value is not None.

        Returns:
        coord - two options:
          n == 1 : tuple of coordinates
          n > 1  : list of tuple of coordinates
        """
        shape = self.signal_np.shape
        coord_list = []
        for idx_chunk in range(n):
            coord = [0, 0, 0]
            for i in range(len(dims_chunk)):
                if dims_pin[i] is None:
                    coord[i] = np.random.random_integers(0, shape[i] - dims_chunk[i])
                else:
                    coord[i] = dims_pin[i]
            coord_list.append(tuple(coord))
        return coord_list if n > 1 else coord_list[0]

    def _extract_chunk(self, dims_chunk, coord):
        """Returns smaller arrays extracted from signal_/target_np.

        Parameters:
        dims_chunk - tuple of chunk dimensions
        coord - tuple to indicate coordinate in larger_ar to start the extraction. If None,
          a random valid coordinate will be selected
        """
        slices = []
        for i in range(len(coord)):
            slices.append(slice(coord[i], coord[i] + dims_chunk[i]))
        return self.signal_np[slices], self.target_np[slices]
    

class CziLoader(Loader):
    def __init__(self, file_path, channel_light, channel_nuclear):
        super().__init__(file_path)
        # Currently, expect to deal only with CZI files where 'B' and '0' dimensions are size 1
        self.czi_reader = io.cziReader.CziReader(file_path)
        czi_np = self.czi_reader.czi.asarray()
        assert (czi_np.shape[0], czi_np.shape[-1]) == (1, 1), \
            "'B' and '0' dimensions are not size 1"
        self.czi_np = czi_np

        # extract light and nuclear channels
        self.signal_np = self.get_volume(channel_light)
        self.target_np = self.get_volume(channel_nuclear)

        self._process_signal_np()
        self._process_target_np()

    def _process_signal_np(self):
        mean = np.mean(self.signal_np)
        std = np.std(self.signal_np)
        self.signal_np = (self.signal_np - mean)/std
    
    def _process_target_np(self):
        # add background subtraction
        mean = np.mean(self.target_np)
        std = np.std(self.target_np)
        self.target_np = (self.target_np - mean)/std
        
    def get_volume(self, c):
        """Returns the image volume for the specified channel."""
        if self.czi_reader.hasTimeDimension:
            raise NotImplementedError  # TODO: handle case of CZI images with T dimension
        if self.czi_reader.czi.axes == b'BCZYX0':
            return self.czi_np[0, c, :, :, :, 0]

def print_array_stats(ar):
    print('shape:', ar.shape, '|', 'dtype:', ar.dtype)
    print('min:', ar.min(), '| max:', ar.max(), '| median', np.median(ar))
        
def pick_random_chunk_coord(shape, dims_chunk, n=1, dims_pin=(None, None, None)):
    """Returns a random coordinate from where an array chunk can be extracted from a larger array.
    
    Parameters:
    shape - tuple indicating shape of larger array
    dims_chunk - tuple of chunk dimensions
    n - (int, optional) - 
    dims_pin (optional) - tuple of fixed coordinate values. Dimensions for the returned coordinate
      will be set to the dims_pin value if the value is not None.
    
    Returns:
    coord - tuple of coordinate within larger array
    """
    assert len(shape) == len(dims_chunk)
    coord_list = []
    for idx_chunk in range(n):
        coord = [0, 0, 0]
        for i in range(len(dims_chunk)):
            if dims_pin[i] is None:
                coord[i] = np.random.random_integers(0, shape[i] - dims_chunk[i])
            else:
                coord[i] = dims_pin[i]
        coord_list.append(tuple(coord))
    return coord_list if n > 1 else coord_list[0]

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

def draw_rect(img, coord_tr, dims_rect, thickness=5):
    """Draw rectangle on image.

    Parameters:
    img - 2d numpy array (image is modified)
    coord_tr - coordinate within img to be top-right corner or rectangle
    dims_rect - 2-value tuple indicated the dimensions of the rectangle

    Returns:
    None
    """
    assert len(img.shape) == len(coord_tr) == len(dims_rect) == 2
    color = 0
    for i in range(thickness):
        if (i+1)*2 <= dims_rect[0]:
            # create horizontal lines
            img[coord_tr[0] + i, coord_tr[1]:coord_tr[1] + dims_rect[1]] = color
            img[coord_tr[0] + dims_rect[0] - 1 - i, coord_tr[1]:coord_tr[1] + dims_rect[1]] = color
        if (i+1)*2 <= dims_rect[1]:
            # create vertical lines
            img[coord_tr[0]:coord_tr[0] + dims_rect[0], coord_tr[1] + i] = color
            img[coord_tr[0]:coord_tr[0] + dims_rect[0], coord_tr[1] + dims_rect[1] - 1 - i] = color
            
# ***** TESTS *****
            
def test_czireader():
    fname = './test_images/20161209_C01_001.czi'
    # fname = './test_images/20161219_C01_034.czi'
    reader = io.cziReader.CziReader(fname)
    czi_np = reader.load()
    print(type(czi_np), czi_np.shape)
    print('axes:', reader.czi.axes)
    print('meta:', reader.czi.metadata)
    meta = reader.czi.metadata
    # pdb.set_trace()
    # print(meta.getroot())

def test_draw_rect():
    img = np.ones((12, 10))*255
    draw_rect(img, (1,2), (5,7))
    print(img)
    img = np.ones((12, 10))*255
    draw_rect(img, (0,0), (7,7))
    print(img)

def test_pick_random_chunk_coord():
    print('*** test_pick_random_chunk_coord ***')
    shape = (50, 1000, 2000)
    dims_chunk = (22, 33, 44)
    result = pick_random_chunk_coord(shape, dims_chunk, n=3, dims_pin=(None, None, None))
    print('result:')
    print(result)

    dims_chunk = (5,6,7)
    n_runs = 10000
    coords_random = np.zeros((n_runs, len(dims_chunk)), dtype=np.int)
    print('generating', n_runs, 'random coordinates for chunk of shape', dims_chunk)
    dims_pin = (None, None, None)
    for i in range(n_runs):
        coord = pick_random_chunk_coord(shape, dims_chunk, dims_pin=dims_pin)
        # print('random coordinate for chunk of size', dims_chunk, '->', coord)
        coords_random[i] = coord
        # test = extract_chunk(vol, dims_chunk, coord)
        # print('extracted chunk shape:', test.shape)
    print('random coord mins:', np.amin(coords_random, axis=0))
    print('random coord maxs:', np.amax(coords_random, axis=0))
    print('first 5 random coords:')
    print(coords_random[:5])
    
def test_CziLoader():
    print('*** test_CziLoader ***')
    fname = './test_images/20161209_C01_001.czi'
    loader = CziLoader(fname, channel_light=3, channel_nuclear=2)
    x, y = loader.get_batch(16, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
    print('x, y shapes:', x.shape, y.shape)
    model = models.Model()

    model.do_train_iter(x, y)
    model.do_train_iter(x, y)
    model.do_train_iter(x, y)

if __name__ == '__main__':
    # test_czireader()
    # test_draw_rect()
    # test_pick_random_chunk_coord()
    test_CziLoader()
