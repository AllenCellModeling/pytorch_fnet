import numpy as np
import pdb

class MultiFileDataProvider(object):
    def __init__(self, dataset, buffer_size, n_iter, batch_size, replace_interval, dims_chunk=(32, 64, 64), dims_pin=(None, None, None)):
        """
        dataset - DataSet instance
        buffer_size - (int) number images to generate batches from
        n_iter - (int) number of batches that the data provider should supply
        ...
        replace_interval - (int) number of batches between buffer item replacements. Set to -1 for no replacement.
        dims_chunk - (tuple) shape of extracted chunks
        dims_pin - (tuple) optionally pin the chunk extraction from this coordinate. Use None to indicate no pinning
                   for any particular dimension.
        """
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._replace_interval = replace_interval

        self.last_sources = ''  # str indicating indices of folders in buffer

        self._dims_chunk = dims_chunk
        self._dims_pin = dims_pin
        
        self._buffer = []
        self._n_folders = len(dataset)
        self._idx_folder = 0
        self._count_iter = 0
        self._idx_replace = 0  # next 

        self._fill_buffer()

    def set_dims_pin(self, dims_pin):
        self._dims_pin = dims_pin

    def get_dims_chunk(self):
        return self._dims_chunk

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
        volumes = None
        while volumes is None:  # TODO: potential for infinite loop here; limit max number of tries?
            volumes = self._dataset[self._idx_folder]
            if volumes:
                idx_folder = self._idx_folder
            self._incr_idx_folder()
        return (idx_folder, volumes[0], volumes[1])

    def _update_last_sources(self):
        source_list = [str(package[0]) for package in self._buffer]
        self.last_sources = '|'.join(source_list)

    def get_batch(self):
        """Get a batch of examples from source data."
        
        Returns:
        batch_x, batch_y - (2 numpy arrays) each array will have shape (n, 1) + dims_chunk.
        """
        shape_batch = (self._batch_size, 1) + self._dims_chunk
        batch_x = np.zeros(shape_batch, dtype=np.float32)
        batch_y = np.zeros(shape_batch, dtype=np.float32)
        coords = self._pick_random_chunk_coords()
        for i in range(len(coords)):
            coord = coords[i]
            chunks_tup = self._extract_chunk(coord)
            # print('  DEBUG: batch element', i, coord, chunks_tup[0].shape, chunks_tup[1].shape)
            batch_x[i, 0, ...] = chunks_tup[0]
            batch_y[i, 0, ...] = chunks_tup[1]
        return (batch_x, batch_y)
    
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
        """Returns smaller arrays extracted from images in buffer.

        Parameters:
        coord - (list) tuples in the form of (idx_buffer, (z, y, z)) to indicate where to extract chunks from buffer
        """
        idx_buf = coord[0]
        coord_img = coord[1]
        slices = []
        for i in range(len(coord_img)):
            slices.append(slice(coord_img[i], coord_img[i] + self._dims_chunk[i]))
        return self._buffer[idx_buf][1][slices], self._buffer[idx_buf][2][slices]

    def __len__(self):
        return self._n_iter
    
    def __iter__(self):
        return self

    def __next__(self):
        if self._count_iter == self._n_iter:
            raise StopIteration
        self._count_iter += 1
        self._update_last_sources()
        if (self._replace_interval > 0) and (self._count_iter % self._replace_interval == 0):
            self._replace_buffer_item()
        return self.get_batch()


DataProvider = MultiFileDataProvider  # to fit with the train_model API

