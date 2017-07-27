import pdb
import numpy as np
from util.misc import pad_mirror

# For each element in a dataset, provides 1 "batch" of size 1.
class WholeImgDataProvider(object):
    def __init__(self, dataset, shape_adj,
                 multiple_of=16):
        """
        Parameter:
        dataset - (DataSet)
        shape_adj - (string or tuple)
        mutliple_of - (int) default: 16
        """
        assert multiple_of in (4, 8, 16, 32)
        self._dataset = dataset
        self._shape_adj = shape_adj
        self._multiple_of = multiple_of

        if self._shape_adj == None:
            self._apply_adjustment = self._do_nothing_adjustment
        elif self._shape_adj == 'crop':
            self._apply_adjustment = self._do_crop_adjustment
        elif self._shape_adj == 'pad_mirror':
            self._apply_adjustment = self._do_pad_mirror_adjustment
        elif self._shape_adj == 'pad_zero':
            self._apply_adjustment = self._do_pad_zero_adjustment
        elif isinstance(self._shape_adj, (tuple, list)):
            self._apply_adjustment = self._do_fixed_dim_adjustment
        else:
            raise NotImplementedError
        
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
        # for each dimension, get next lower number divisible by multiple_of
        shape_new = (1, 1,) + tuple((i & ~(self._multiple_of - 1)) for i in shape_vol)
        # print('DEBUG:', shape_vol, shape_new)
        data = []
        for vol in volumes_tuple:
            data.append(np.zeros(shape_new, dtype=np.float32))
            data[-1][0, 0, ] = vol[:shape_new[2], :shape_new[3], :shape_new[4]]
        return tuple(data)
    
    def _do_pad_zero_adjustment(self, volumes_tuple):
        shape_vol = volumes_tuple[0].shape
        # for each dimension, get next higher number divisible by multiple_of
        # example calculations:
        # (15 + 15) & ~0xF = 16
        # (32 + 15) & ~0xF = 32
        # (33 + 15) & ~0xF = 48
        shape_new = (1, 1,) + tuple(((i + (self._multiple_of - 1)) & ~(self._multiple_of - 1)) for i in shape_vol)
        data = []
        for vol in volumes_tuple:
            data.append(np.zeros(shape_new, dtype=np.float32))
            slices = []
            for i in range(len(shape_vol)):
                start = shape_new[i + 2]//2 - shape_vol[i]//2
                slices.append(slice(start, start + shape_vol[i]))
            data[-1][0, 0, slices[0], slices[1], slices[2]] = vol
        # print('DEBUG:', shape_vol, shape_new, slices)
        return tuple(data)

    def _do_pad_mirror_adjustment(self, volumes_tuple):
        shape_vol = volumes_tuple[0].shape
        # for each dimension, get next higher number divisible by multiple_of
        # example calculations:
        # (15 + 15) & ~0xF = 16
        # (32 + 15) & ~0xF = 32
        # (33 + 15) & ~0xF = 48
        shape_new = (1, 1,) + tuple(((i + (self._multiple_of - 1)) & ~(self._multiple_of - 1)) for i in shape_vol)
        padding = tuple(np.ceil( tuple((shape_new[2 + i] - shape_vol[i])/2 for i in range(len(shape_vol)))).astype(int))
        data = []
        for vol in volumes_tuple:
            padded = pad_mirror(vol, padding)
            # for each dim, padded array might be 1 larger than necessary if starting dim size was an odd number.
            data.append(np.zeros(shape_new, dtype=np.float32))
            data[-1][0, 0, ] = padded[:shape_new[2], :shape_new[3], :shape_new[4]]
        # print('DEBUG:', shape_vol, shape_new, padding, padded.shape)
        return tuple(data)

    def _do_fixed_dim_adjustment(self, volumes_tuple):
        shape_vol = volumes_tuple[0].shape
        shape_new = (1, 1, ) + tuple(self._shape_adj)
        assert len(shape_vol) == (len(shape_new) - 2)
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
