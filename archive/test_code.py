class Loader(object):
    def __init__(self):
        self.vol_light_np = None
        self.vol_nuc_np = None

    def resize_data(self, factors):
        """Rescale the light/nuclear channels.

        Parameters:
        factors - tuple of scaling factors for each dimension.
        """
        self.vol_light_np = proc.resize(self.vol_light_np, factors);
        self.vol_nuc_np = proc.resize(self.vol_nuc_np, factors);

    def get_batch(self, n, dims_chunk=(32, 32, 32), dims_pin=(None, None, None), return_coords=False):
        """Get a batch of examples from source data."

        Parameters:
        n - (int) batch size.
        dims_chunk - (tuple) ZYX dimensions of each example.
        return_coords - (boolean) if True, also return the coordinates from where chunks were taken.
        
        Returns:
        return_coord == False
        batch_x, batch_y - (2 numpy arrays) each array will have shape (n, 1) + dims_chunk.

        return_coord == True
        batch_x, batch_y, coords - same as above but with the addition of the chunk coordinates.
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
        return (batch_x, batch_y) if not return_coords else (batch_x, batch_y, coords)

    def _pick_random_chunk_coord(self, dims_chunk, n=1, dims_pin=(None, None, None)):
        """Returns a random coordinate from where an array chunk can be extracted from signal_, vol_nuc_np.

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
        shape = self.vol_light_np.shape
        coord_list = []
        for idx_chunk in range(n):
            coord = [0, 0, 0]
            for i in range(len(dims_chunk)):
                if dims_pin[i] is None:
                    # coord[i] = np.random.random_integers(0, shape[i] - dims_chunk[i])
                    coord[i] = np.random.randint(0, shape[i] - dims_chunk[i] + 1)
                else:
                    coord[i] = dims_pin[i]
            coord_list.append(tuple(coord))
        return coord_list if n > 1 else coord_list[0]

    def _extract_chunk(self, dims_chunk, coord):
        """Returns smaller arrays extracted from signal_/vol_nuc_np.

        Parameters:
        dims_chunk - tuple of chunk dimensions
        coord - tuple to indicate coordinate in larger_ar to start the extraction. If None,
          a random valid coordinate will be selected
        """
        slices = []
        for i in range(len(coord)):
            slices.append(slice(coord[i], coord[i] + dims_chunk[i]))
        return self.vol_light_np[slices], self.vol_nuc_np[slices]
    

class TifLoader(Loader):
    def __init__(self, file_path_light, file_path_nuc):
        super().__init__()
        

        
class CziLoader(Loader):
    def __init__(self, file_path, channel_light, channel_nuclear):
        super().__init__()
        # Currently, expect to deal only with CZI files where 'B' and '0' dimensions are size 1
        self.czi_reader = io.cziReader.CziReader(file_path)
        czi_np = self.czi_reader.czi.asarray()
        assert (czi_np.shape[0], czi_np.shape[-1]) == (1, 1), \
            "'B' and '0' dimensions are not size 1"
        self.czi_np = czi_np

        # extract light and nuclear channels
        self.vol_light_np = self.get_volume(channel_light)
        self.vol_nuc_np = self.get_volume(channel_nuclear)

        z_fac = 0.96
        xy_fac = 0.22
        factors = (z_fac, xy_fac, xy_fac)
        self.resize_data(factors)
        self._process_vol_light_np()
        self._process_vol_nuc_np()

    def _process_vol_light_np(self):
        # mean = np.mean(self.vol_light_np)
        # std = np.std(self.vol_light_np)
        # self.vol_light_np = (self.vol_light_np - mean)/std
        self.vol_light_np = self.vol_light_np/np.amax(self.vol_light_np)
    
    def _process_vol_nuc_np(self):
        # self.vol_nuc_np[self.vol_nuc_np < np.median(self.vol_nuc_np)] = 0
        # mean = np.mean(self.vol_nuc_np)
        # std = np.std(self.vol_nuc_np)
        # self.vol_nuc_np = (self.vol_nuc_np - mean)/std
        self.vol_nuc_np = self.vol_nuc_np/np.amax(self.vol_nuc_np)
        
    def get_volume(self, c):
        """Returns the image volume for the specified channel."""
        if self.czi_reader.hasTimeDimension:
            raise NotImplementedError  # TODO: handle case of CZI images with T dimension
        if self.czi_reader.czi.axes == b'BCZYX0':
            return self.czi_np[0, c, :, :, :, 0]

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

def test_resize(show_figures=False):
    print('*** test_resize ***')
    fname = './test_images/20161209_C01_001.czi'
    loader = CziLoader(fname, channel_light=3, channel_nuclear=2)
    n_chunks = 3
    dims_chunk = (32, 64, 64)
    z_max_before = find_z_of_max_slice(loader.vol_nuc_np)
    z_pin_before = z_max_before - dims_chunk[0]//2
    if z_pin_before< 0:
        z_pin_before = 0
    print('before resize:')
    print('  signal, target shapes:', loader.vol_light_np.shape, loader.vol_nuc_np.shape)
    print('  target max z:', z_max_before)
    batch_before = loader.get_batch(n_chunks, dims_chunk=dims_chunk, dims_pin=(z_pin_before, None, None), return_coords=True)
    if show_figures:
        display_batch(loader.vol_light_np, loader.vol_nuc_np, batch_before)
    
    z_max_after = find_z_of_max_slice(loader.vol_nuc_np)
    z_pin_after = z_max_after - dims_chunk[0]//2
    if z_pin_after < 0:
        z_pin_after = 0
    print('after resize by factors:', factors)
    print('  signal, target shapes:', loader.vol_light_np.shape, loader.vol_nuc_np.shape)
    print('  target max z:', z_max_after)
    # get and display random chunks
    batch_after = loader.get_batch(n_chunks, dims_chunk=dims_chunk, dims_pin=(z_pin_after, None, None), return_coords=True)
    if show_figures:
        display_batch(loader.vol_light_np, loader.vol_nuc_np, batch_after)

def test_find_z_of_max_slice():
    print('*** test_find_max_target_z ***')
    fname = './test_images/20161209_C01_001.czi'
    loader = CziLoader(fname, channel_light=3, channel_nuclear=2)
    z_max = find_z_of_max_slice(loader.vol_nuc_np)
    print('z of vol_nuc_np with max fluorescence:', z_max)

def test_TifLoader():
    print('*** test_TifLoader ***')
    fname_light = '/allen/aics/modeling/cheko/projects/nucleus_predictor/test_images/20161209_C01_021.czi/20161209_C01_021.czi_1_trans.tif'
    fname_nuc = '/allen/aics/modeling/cheko/projects/nucleus_predictor/test_images/20161209_C01_021.czi/20161209_C01_021.czi_1_dna.tif'
    loader = TifLoader(fname_light, fname_nuc)
    print(loader)
    
def train_eval():
    fname = './test_images/20161209_C01_001.czi'
    loader = CziLoader(fname, channel_light=3, channel_nuclear=2)
    print_array_stats(loader.vol_light_np)
    print_array_stats(loader.vol_nuc_np)

    np.random.seed(666)
    # x, y = loader.get_batch(16, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
    model = models.Model(mult_chan=32, depth=4)
    n_train_iter = 10
    for i in range(n_train_iter):
        x, y = loader.get_batch(16, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
        model.do_train_iter(x, y)

    n_check = 10  # number of examples to check
    x_val = x[:n_check]
    y_true = y[:n_check]
    y_pred = model.predict(x_val)
    display_visual_eval_images(x_val, y_true, y_pred)

if __name__ == '__main__':
    # test_czireader()
    # test_draw_rect()
    # test_pick_random_chunk_coord()
    # test_CziLoader()
    # test_find_z_of_max_slice()
    # test_resize()
    # test_TifLoader()
    train_eval()
