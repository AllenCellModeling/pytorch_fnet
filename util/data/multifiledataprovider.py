import numpy as np
import queue

class MultiFileDataProvider(object):
    def __init__(self, dataset, buffer_size, n_iter, batch_size):
        """
        dataset - DataSet instance
        buffer_size - (int) number images to generate batches from
        n_iter - (int) number of batches that the data provider should supply
        """
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._n_iter = n_iter
        self._batch_size = batch_size

        # TODO: make parameters
        self._dims_chunk = (32, 64, 64)
        self._dims_pin=(None, None, None)
        
        self._buffer = []
        self._n_folders = len(dataset)
        self._idx_folder = 0
        self._source_list = []
        self._count_iter = 0

        self._fill_buffer()
        print('DEBUG: source list =>', self.get_sources())

    def _fill_buffer(self):
        while len(self._buffer) < self._buffer_size:
            package = self._create_package()
            self._buffer.append(package)
            self._source_list.append(str(package[0]))
        print('DEBUG: current buffer size', len(self._buffer))

    def _incr_idx_folder(self):
        self._idx_folder += 1
        if self._idx_folder % self._n_folders == 0:
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

    def _enqueue_fifo(self, package):
        """Enqueue item to fifo.
        package - 3-element tuple (idx_folder, vol_trans, vol_dna)
        """
        self._fifo.put(package)

    def get_sources(self):
        return '|'.join(self._source_list)

    def get_batch(self):
        """Get a batch of examples from source data."

        Parameters:
        return_coords - (boolean) if True, also return the coordinates from where chunks were taken.
        
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
            # print(i, coord, chunks_tup[0].shape, chunks_tup[1].shape)
            batch_x[i, 0, ...] = chunks_tup[0]
            batch_y[i, 0, ...] = chunks_tup[1]
        return (batch_x, batch_y)
    
    def _pick_random_chunk_coords(self):
        """Returns a random coordinate from random images in buffer.

        Returns:
        coords - list of tuple of coordinates
        """
        idx_rand = np.random.randint(0, self._buffer_size)
        shape = self._buffer[idx_rand][1].shape  # get shape from trans channel image
        coord_list = []
        for idx_chunk in range(self._batch_size):
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
        return self.get_batch()

