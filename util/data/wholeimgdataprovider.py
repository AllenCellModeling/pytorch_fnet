import pdb
import numpy as np

class WholeImgDataProvider(object):
    def __init__(self, dataset, shape_adj):
        assert shape_adj in (None, 'crop', 'mirror', 'zero_pad')
        self._dataset = dataset
        self._shape_adj = shape_adj

        if self._shape_adj == None:
            self._apply_adjustment = self._do_nothing_adjustment
        elif self._shape_adj == 'crop':
            self._apply_adjustment = self._do_crop_adjustment
        
        self._count_iter = 0
        self._n_iter = len(dataset)

    def _do_nothing_adjustment(self, volumes_tuple):
        shape = (1, 1, ) + volumes_tuple[0].shape
        data = []
        for vol in volumes_tuple:
            data.append(np.zeros(shape, dtype=np.float32))
            data[-1][0, 0, ] = vol
        return tuple(data)

    def _do_crop_adjustment(self, volumes_tuple):
        shape_vol = volumes_tuple[0].shape
        # for each dimension, get next lower number divisible by 16
        shape_new = (1, 1,) + tuple((i >> 4 << 4) for i in shape_vol)
        print('DEBUG:', shape_vol, shape_new)
        data = []
        for vol in volumes_tuple:
            data.append(np.zeros(shape_new, dtype=np.float32))
            data[-1][0, 0, ] = vol[:shape_new[2], :shape_new[3], :shape_new[4]]
        return tuple(data)

    def __len__(self):
        return len(dataset)

    def __iter__(self):
        self._count_iter = 0
        return self

    def __next__(self):
        if self._count_iter >= self._n_iter:
            raise StopIteration
        volumes_tuple = self._dataset[self._count_iter]
        data_tuple = self._apply_adjustment(volumes_tuple)
        self._count_iter += 1
        return data_tuple
