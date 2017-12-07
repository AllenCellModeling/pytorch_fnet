import numpy as np
import warnings
from fnet import get_vol_transformed
import pdb

class ChunkDataProvider(object):
    def __init__(self, dataset, buffer_size, batch_size, replace_interval,
                 dims_chunk=(32, 64, 64), dims_pin=(None, None, None),
                 transforms=None,
                 choices_augmentation = None,
    ):
        """
        dataset - DataSet instance
        buffer_size - (int) number images to generate batches from
        ...
        replace_interval - (int) number of batches between buffer item replacements. Set to -1 for no replacement.
        dims_chunk - (tuple) shape of extracted chunks
        dims_pin - (tuple) optionally pin the chunk extraction from this coordinate. Use None to indicate no pinning
                   for any particular dimension.
        transforms - list of transforms to apply to each DataSet element
        """
        assert transforms is None or isinstance(transforms, (list, tuple))
        assert choices_augmentation is None or all(i in range(8) for i in choices_augmentation)
        print('DEBUG: augmentation', choices_augmentation)
        
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._replace_interval = replace_interval
        self._transforms = transforms

        self.last_sources = ''  # str indicating indices of folders in buffer

        self._dims_chunk = dims_chunk
        self._dims_pin = dims_pin
        
        self._buffer = []
        self._n_folders = len(dataset)
        self._idx_folder = 0
        self._count_iter = 0
        self._idx_replace = 0  # next 

        self._fill_buffer()
        self._update_last_sources()
        self._shape_batch = [self._batch_size, 1] + list(self._dims_chunk)
        self._dims_chunk_options = (self._dims_chunk, (self._dims_chunk[0]//2, *self._dims_chunk[1:]))
        
        self.choices_augmentation = choices_augmentation

    def use_test_set(self):
        self._dataset.use_test_set()
        
    def use_train_set(self):
        self._dataset.use_train_set()

    def set_dims_pin(self, dims_pin):
        self._dims_pin = dims_pin

    def get_dims_chunk(self):
        return self._dims_chunk

    def _vol_size_okay(self, vol):
        return all(vol.shape[i] >= self._dims_chunk[i] for i in range(vol.ndim))

    def _replace_buffer_item(self):
        """Replace oldest package in buffer with another package."""
        package = self._create_package()
        self._buffer[self._idx_replace] = package
        self._idx_replace += 1
        if self._idx_replace >= self._buffer_size:
            self._idx_replace = 0

    def _fill_buffer(self):
        while len(self._buffer) < self._buffer_size:
            package = self._create_package()
            self._buffer.append(package)

    def _incr_idx_folder(self):
        self._idx_folder += 1
        if self._idx_folder >= self._n_folders:
            self._idx_folder = 0

    def _create_package(self):
        """Read signal, target images from current folder and return data package.

        Returns:
        package - 3-element tuple (idx_folder, vol_signal, vol_target)
        """
        tries = 5
        volumes = None
        while volumes is None and tries > 0:
            volumes = self._dataset[self._idx_folder]
            if volumes:
                if self._vol_size_okay(volumes[0]):
                    idx_folder = self._idx_folder
                else:
                    warnings.warn('bad size: {}. skipping....'.format(volumes[0].shape))
                    volumes = None
            self._incr_idx_folder()
            tries -= 1
        if tries <= 0:
            raise
        return (idx_folder, volumes[0], volumes[1])

    def _update_last_sources(self):
        source_list = [str(package[0]) for package in self._buffer]
        self.last_sources = '|'.join(source_list)

    def _augment_chunks(self, chunks):
        if self.choices_augmentation is None:
            return chunks
        chunks_new = []
        choice = np.random.choice(self.choices_augmentation)
        for chunk in chunks:
            chunk_new = chunk
            if choice in [1, 3, 5, 7]:
                chunk_new = np.flip(chunk_new, axis=1)
            if   choice in [2, 3]:
                chunk_new = np.rot90(chunk_new, 1, axes=(1, 2))
            elif choice in [4, 5]:
                chunk_new = np.rot90(chunk_new, 2, axes=(1, 2))
            elif choice in [6, 7]:
                chunk_new = np.rot90(chunk_new, 3, axes=(1, 2))
            chunks_new.append(chunk_new)
        return chunks_new

    def _gen_batch(self):
        """Generate a batch from sources in self._buffer
        
        Returns:
        batch_x, batch_y - (2 numpy arrays) each array will have shape (n, 1) + dims_chunk.
        """
        batch_x, batch_y = None, None
        coords = self._pick_random_chunk_coords()
        for i in range(len(coords)):
            coord = coords[i]
            chunks_tup = self._extract_chunk(coord)
            if self._transforms is not None:
                chunks_transformed = []
                for j, transform in enumerate(self._transforms):
                    chunks_transformed.append(get_vol_transformed(chunks_tup[j], self._transforms[j]))
            else:
                chunks_transformed = chunks_tup
            if batch_x is None or batch_y is None:
                batch_x = np.zeros((self._batch_size, 1, ) + chunks_transformed[0].shape, dtype=np.float32)
                batch_y = np.zeros((self._batch_size, 1, ) + chunks_transformed[1].shape, dtype=np.float32)
            # pdb.set_trace()
            chunks_augmented = self._augment_chunks(chunks_transformed)
            batch_x[i, 0, ...] = chunks_augmented[0]
            batch_y[i, 0, ...] = chunks_augmented[1]
        return batch_x, batch_y
    
    def _pick_random_chunk_coords(self):
        """Returns a random coordinate from random images in buffer.

        Returns:
        coords - list of tuples of the form (idx_buffer, (z, y, z))
        """
        coord_list = []
        for idx_chunk in range(self._batch_size):
            idx_rand = np.random.randint(0, self._buffer_size)
            shape = self._buffer[idx_rand][1].shape  # get shape from trans channel image
            coord_3d = [0]*len(self._dims_chunk)
            for i in range(len(coord_3d)):
                if self._dims_pin[i] is None:
                    coord_3d[i] = np.random.randint(0, shape[i] - self._dims_chunk[i] + 1) # upper bound of randint is exclusive, so +1
                else:
                    coord_3d[i] = self._dims_pin[i]
            coord_list.append((idx_rand, tuple(coord_3d)))
        return coord_list

    def _extract_chunk(self, coord):
        """Returns arrays extracted from images in buffer.

        Parameters:
        coord - (list) tuples in the form of (idx_buffer, (z, y, z)) to indicate where to extract chunks from buffer
        """
        idx_buf = coord[0]
        coord_img = coord[1]
        slices = []
        for i in range(len(coord_img)):
            slices.append(slice(coord_img[i], coord_img[i] + self._dims_chunk[i]))
        return self._buffer[idx_buf][1][slices], self._buffer[idx_buf][2][slices]

    def get_batch(self):
        """Get a batch of examples from source data."
        
        Returns:
        batch_x, batch_y - (2 numpy arrays) each array will have shape (n, 1) + dims_chunk.
        """
        self._count_iter += 1
        if (self._replace_interval > 0) and (self._count_iter % self._replace_interval == 0):
            self._replace_buffer_item()
            self._update_last_sources()
        return self._gen_batch()
