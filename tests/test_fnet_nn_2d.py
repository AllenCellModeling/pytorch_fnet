import unittest
import torch.utils.data
import numpy as np
import fnet.nn_modules.fnet_nn_2d as nn_module
from fnet.data import DummyChunkDataset
import pdb

class Test2D(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        dims_chunk = (1, 16, 32)
        self.ds = DummyChunkDataset(dims_chunk)
        self.dl = torch.utils.data.DataLoader(
            self.ds,
            batch_size = 2,
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
        self.assertAlmostEqual(396.2701721191406, loss.data[0])

if __name__ == '__main__':
    unittest.main()
