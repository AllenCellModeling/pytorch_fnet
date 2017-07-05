import numpy as np
import aicsimage.processing as proc
import pdb

class DataProvider(object):
    def __init__(self):
        self.vol_trans_np = None
        self.vol_dna_np = None
        self.batch_size = None

    def resize_data(self, factors):
        """Resize the transmitted light/DNA channels.

        Parameters:
        factors - tuple of scaling factors for each dimension.
        """
        self.vol_trans_np = proc.resize(self.vol_trans_np, factors)
        self.vol_dna_np = proc.resize(self.vol_dna_np, factors)

    def get_batch(self, batch_size=None,
                  dims_chunk=(32, 64, 64), dims_pin=(None, None, None), return_coords=False):
        """Get a batch of examples from source data."

        Parameters:
        dims_chunk - (tuple) ZYX dimensions of each example.
        return_coords - (boolean) if True, also return the coordinates from where chunks were taken.
        
        Returns:
        return_coord == False
        batch_x, batch_y - (2 numpy arrays) each array will have shape (n, 1) + dims_chunk.

        return_coord == True
        batch_x, batch_y, coords - same as above but with the addition of the chunk coordinates.
        """
        if batch_size is None:
            batch_size = self.batch_size
        shape_batch = (batch_size, 1) + dims_chunk
        batch_x = np.zeros(shape_batch, dtype=np.float32)
        batch_y = np.zeros(shape_batch, dtype=np.float32)
        coords = self._pick_random_chunk_coord(dims_chunk, n=batch_size, dims_pin=dims_pin)
        for i in range(len(coords)):
            coord = coords[i]
            # print(coord)
            chunks_tup = self._extract_chunk(dims_chunk, coord)
            batch_x[i, 0, ...] = chunks_tup[0]
            batch_y[i, 0, ...] = chunks_tup[1]
        return (batch_x, batch_y) if not return_coords else (batch_x, batch_y, coords)

    def _pick_random_chunk_coord(self, dims_chunk, n=1, dims_pin=(None, None, None)):
        """Returns a random coordinate from where an array chunk can be extracted from signal_, vol_dna_np.

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
        shape = self.vol_trans_np.shape
        coord_list = []
        for idx_chunk in range(n):
            coord = [0, 0, 0]
            for i in range(len(dims_chunk)):
                if dims_pin[i] is None:
                    coord[i] = np.random.randint(0, shape[i] - dims_chunk[i] + 1)
                else:
                    coord[i] = dims_pin[i]
            coord_list.append(tuple(coord))
        return coord_list if n > 1 else coord_list[0]

    def _extract_chunk(self, dims_chunk, coord):
        """Returns smaller arrays extracted from signal_/vol_dna_np.

        Parameters:
        dims_chunk - tuple of chunk dimensions
        coord - tuple to indicate coordinate in larger_ar to start the extraction. If None,
          a random valid coordinate will be selected
        """
        slices = []
        for i in range(len(coord)):
            slices.append(slice(coord[i], coord[i] + dims_chunk[i]))
        return self.vol_trans_np[slices], self.vol_dna_np[slices]


if __name__ == '__main__':
    data = DataProvider()
    print('done')
