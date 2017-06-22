import aicsimage.processing as proc
import aicsimage.io as io
import numpy as np
import models
import matplotlib.pyplot as plt
import pdb

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

def show_img(ar):
    import PIL
    import PIL.ImageOps
    from IPython.core.display import display
    img_norm = ar - ar.min()
    img_norm *= 255./img_norm.max()
    img_pil = PIL.Image.fromarray(img_norm).convert('L')
    display(img_pil)
    
def find_z_of_max_slice(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice with the most signal."""
    z_max = np.argmax(np.sum(ar, axis=(1, 2)))
    return z_max
        
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

def draw_rect(img, coord_tl, dims_rect, thickness=3, color=0):
    """Draw rectangle on image.

    Parameters:
    img - 2d numpy array (image is modified)
    coord_tl - coordinate within img to be top-left corner or rectangle
    dims_rect - 2-value tuple indicated the dimensions of the rectangle

    Returns:
    None
    """
    assert len(img.shape) == len(coord_tl) == len(dims_rect) == 2
    for i in range(thickness):
        if (i+1)*2 <= dims_rect[0]:
            # create horizontal lines
            img[coord_tl[0] + i, coord_tl[1]:coord_tl[1] + dims_rect[1]] = color
            img[coord_tl[0] + dims_rect[0] - 1 - i, coord_tl[1]:coord_tl[1] + dims_rect[1]] = color
        if (i+1)*2 <= dims_rect[1]:
            # create vertical lines
            img[coord_tl[0]:coord_tl[0] + dims_rect[0], coord_tl[1] + i] = color
            img[coord_tl[0]:coord_tl[0] + dims_rect[0], coord_tl[1] + dims_rect[1] - 1 - i] = color

def display_batch(vol_light_np, vol_nuc_np, batch):
    """Display images of examples from batch.
    vol_light_np - numpy array
    vol_nuc_np - numpy array
    batch - 3-element tuple: chunks from vol_light_np, chunks from vol_nuc_np, coordinates of chunks
    """
    n_chunks = batch[0].shape[0]
    z_max_big = find_z_of_max_slice(vol_nuc_np)
    img_light, img_nuc = vol_light_np[z_max_big], vol_nuc_np[z_max_big]
    chunk_coord_list = batch[2]
    dims_rect = batch[0].shape[-2:]  # get size of chunk in along yz plane
    min_light, max_light = np.amin(vol_light_np), np.amax(vol_light_np)
    min_nuc, max_nuc = np.amin(vol_nuc_np), np.amax(vol_nuc_np)
    for i in range(len(chunk_coord_list)):
        coord = chunk_coord_list[i][1:]  # get yx coordinates
        draw_rect(img_light, coord, dims_rect, thickness=2, color=min_light)
        draw_rect(img_nuc, coord, dims_rect, thickness=2, color=min_nuc)

    # display originals
    # fig = plt.figure(figsize=(12, 6))
    # fig.suptitle('slice at z: ' + str(z_max_big))
    # ax = fig.add_subplot(121)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.imshow(img_light, cmap='gray', interpolation='bilinear', vmin=-3, vmax=3)
    # ax = fig.add_subplot(122)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.imshow(img_nuc, cmap='gray', interpolation='bilinear', vmin=-3, vmax=3)
    # plt.show()

    print('light volume slice | z =', z_max_big)
    show_img(img_light)
    print('-----')
    print('nuc volume slice | z =', z_max_big)
    show_img(img_nuc)

    # display chunks
    z_mid_chunk = batch[0].shape[2]//2  # z-dim
    for i in range(n_chunks):
        title_str = 'chunk: ' + str(i) + ' | z:' + str(z_mid_chunk)
        fig = plt.figure(figsize=(9, 4))
        fig.suptitle(title_str)
        img_chunk_sig = batch[0][i, 0, z_mid_chunk, ]
        img_chunk_tar = batch[1][i, 0, z_mid_chunk, ]
        ax = fig.add_subplot(1, 2, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img_chunk_sig, cmap='gray', interpolation='bilinear')
        ax = fig.add_subplot(1, 2, 2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img_chunk_tar, cmap='gray', interpolation='bilinear')
        plt.show()
    
            
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

def display_visual_eval_images(signal, target, prediction):
    """Display 3 images: light, nuclear, predicted nuclear.

    Parameters:
    signal (5d numpy array)
    target (5d numpy array)
    prediction (5d numpy array)
    """
    n_examples = signal.shape[0]
    print('Displaying chunk slices for', n_examples, 'examples')
    source_list = [signal, target, prediction]
    z_mid = signal.shape[2]//2
    for ex in range(n_examples):
        fig = plt.figure(figsize=(10, 3))
        for i in range(3):
            fig.suptitle('example: ' + str(ex))
            img = source_list[i][ex, 0, z_mid, ]
            ax = fig.add_subplot(1, 3, i + 1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.imshow(img, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
            ax.imshow(img, cmap='gray', interpolation='bilinear')
    plt.show()

if __name__ == '__main__':
    # test_czireader()
    # test_draw_rect()
    # test_pick_random_chunk_coord()
    # test_CziLoader()
    # test_find_z_of_max_slice()
    # test_resize()
    # test_TifLoader()
    train_eval()
