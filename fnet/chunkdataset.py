from fnet.fnetdataset import FnetDataset
import numpy as np
import pdb

class ChunkDatasetDummy(FnetDataset):
    """Dummy ChunkDataset"""

    def __init__(
            self,
            dataset: FnetDataset,
            dims_chunk,
            random_seed: int = 666,
    ):
        self.dims_chunk = dims_chunk
        self._rng = np.random.RandomState(random_seed)
        self._length = 1234
        self._chunks_signal = 10*self._rng.randn(self._length, *dims_chunk)
        self._chunks_target = 2*self._chunks_signal + 3*self._rng.randn(self._length, *dims_chunk)

    def __getitem__(self, index):
        return (self._chunks_signal[index], self._chunks_target[index])

    def __len__(self):
        return len(self._chunks_signal)

class ChunkDataset(FnetDataset):
    """Dataset that provides chunks/patchs from another dataset."""

    def __init__(self, dataset: FnetDataset):
        pass

    
def _test():
    # dims_chunk = (2,3,4)
    dims_chunk = (4,5)
    ds_test = ChunkDatasetDummy(
        None,
        dims_chunk = dims_chunk,
    )
    print('Dataset len', len(ds_test))
    for i in range(3):
        print('***** {} *****'.format(i))
        element = ds_test[i]
        print(element[0])
        print(element[1])
    
if __name__ == '__main__':
    _test()
