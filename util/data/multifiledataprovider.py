import numpy as np
import queue

class MultiFileDataProvider(object):
    def __init__(self, dataset, buffer_size, n_iter, batch_size, replace_interval, dims_chunk=(32, 64, 64)):
        """
        dataset - DataSet instance
        buffer_size - (int) number images to generate batches from
        n_iter - (int) number of batches that the data provider should supply
        
        replace_interval - (int) number of batches between buffer item replacements. Set to -1 for no replacement.
        """
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._replace_interval = replace_interval

        self.last_sources = ''  # str indicating indices of folders in buffer

        self._dims_chunk = dims_chunk
        self._dims_pin = (None, None, None)
        
        self._buffer = []
        self._n_folders = len(dataset)
        self._idx_folder = 0
        self._count_iter = 0
        self._idx_replace = 0  # next 

        self._fill_buffer()

    def set_dims_pin(self, dims_pin):
        self._dims_pin = dims_pin

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
        print('DEBUG: current buffer size', len(self._buffer))

    def _incr_idx_folder(self):
        self._idx_folder += 1
        if self._idx_folder >= self._n_folders:
            self._idx_folder = 0

    def _create_package(self):
        """Read trans and dna channel images from current folder and return package.

        Returns:
        package - 3-element tuple (idx_folder, vol_trans, vol_dna)
        """
        volumes = None
        while volumes is None:
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
            # print('  DEBUG: get_batch', i, coord, chunks_tup[0].shape, chunks_tup[1].shape)
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
            coord = [0, 0, 0]
            for i in range(len(coord)):
                coord[i] = np.random.randint(0, shape[i] - self._dims_chunk[i] + 1)
            coord_list.append((idx_rand, tuple(coord)))
        return coord_list

    def _extract_chunk(self, coord):
        """Returns smaller arrays extracted from images in buffer.

        Parameters:
        coord - tuple to indicate coordinate in larger_ar to start the extraction. If None,
        """
        idx_buf = coord[0]
        coord_img = coord[1]
        slices = []
        for i in range(len(coord_img)):
            slices.append(slice(coord_img[i], coord_img[i] + self._dims_chunk[i]))
        return self._buffer[idx_buf][1][slices], self._buffer[idx_buf][2][slices]
    
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

