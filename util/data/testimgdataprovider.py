import util.data

class TestImgDataProvider(object):
    """Wrapper class to provider larger chunks for model testing."""
    def __init__(self, dataset, dims_chunk=(32, 128, 128)):
        buffer_size = 1
        self._n_iter = len(dataset)
        batch_size = 1
        self._data = util.data.MultiFileDataProvider(dataset, 1, self._n_iter, batch_size, dims_chunk=dims_chunk)
        print('hi:', self._n_iter, 'elements')

        self._count_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._count_iter == self._n_iter:
            raise StopIteration
        self._count_iter += 1
        return self._data.get_batch()

