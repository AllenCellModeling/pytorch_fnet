from fnet.data.fnetdataset import FnetDataset
import torch
import numpy as np

class DummyChunkDataset(FnetDataset):
    """Dummy dataset to generate random chunks."""

    def __init__(
            self,
            dims_chunk: tuple = (1, 16, 32, 64),
            random_seed: int = 0,
            **kwargs
    ):
        self.dims_chunk = dims_chunk
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)
        self._length = 10
        self._chunks_signal = 10*self._rng.randn(self._length, *dims_chunk)
        self._chunks_target = 2*self._chunks_signal + 3*self._rng.randn(self._length, *dims_chunk)

    def __getitem__(self, index):
        return [torch.Tensor(self._chunks_signal[index]), torch.Tensor(self._chunks_target[index])]

    def __len__(self):
        return len(self._chunks_signal)

    def __repr__(self):
        return 'DummyChunkDataset({}, {})'.format(self.dims_chunk, self.random_seed)


if __name__ == '__main__':
    dims_chunk = (1, 22, 33, 44)
    ds = DummyChunkDataset(
        dims_chunk = dims_chunk,
    )
    print(ds)
    for i in range(3):
        x, y = ds[i]
        print(i, x.shape, y.shape)
        assert x.shape == dims_chunk
        assert y.shape == dims_chunk
    
    
