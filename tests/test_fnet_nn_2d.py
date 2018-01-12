import unittest
import torch.utils.data
import numpy as np
import model_modules.nn_modules.fnet_nn_2d as nn_module
import pdb

class ChunkDatasetDummy(torch.utils.data.Dataset):
    """Dummy ChunkDataset"""

    def __init__(
            self,
            dims_chunk,
            random_seed: int = 0,
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


class Test2D(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        dims_chunk = (1, 16, 32)
        self.ds = ChunkDatasetDummy(dims_chunk)
        self.dl = torch.utils.data.DataLoader(
            self.ds,
            batch_size = 5,
        )
        self.net = nn_module.Net()
        # self.net.cuda(0)

    def test_0(self):
        count = 0
        self.assertEqual(1, 1)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        for itx, batch in enumerate(self.dl):
            batch_x = torch.autograd.Variable(batch[0].float())
            batch_y = torch.autograd.Variable(batch[1].float())
            # print(batch_x.size(), batch_y.size(), batch_x.data.type())
            optimizer.zero_grad()
            batch_pred_y = self.net(batch_x)
            loss = criterion(batch_pred_y, batch_y)
            print('iteration: {:02d} | loss: {:.2f}'.format(itx, loss.data[0]))
            loss.backward()
            optimizer.step()
            # print(batch_pred_y.size())
            count += 1
            if count == 25:
                break
        self.assertAlmostEqual(326.6689147949219, loss.data[0])

if __name__ == '__main__':
    unittest.main()
