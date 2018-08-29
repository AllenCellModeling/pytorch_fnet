import unittest
import torch.utils.data
import numpy as np
import fnet.nn_modules.fnet_nn_2d as nn_module
from fnet.data import DummyChunkDataset
import pdb

class Test2D(unittest.TestCase):
    def setUp(self):
        dims_chunk = (1, 16, 32)
        self.ds = DummyChunkDataset(dims_chunk)
        self.dl = torch.utils.data.DataLoader(
            self.ds,
            batch_size = 2,
        )
        self.net = nn_module.Net()

    def test_train(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        loss_epoch = np.inf
        for epoch in range(2):
            loss_epoch_prev = loss_epoch
            loss_epoch = 0
            for itx, (batch_x, batch_y) in enumerate(self.dl):
                optimizer.zero_grad()
                batch_pred_y = self.net(batch_x)
                loss = criterion(batch_pred_y, batch_y)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            loss_epoch /= len(self.dl)
            self.assertGreater(loss_epoch_prev, loss_epoch)

if __name__ == '__main__':
    unittest.main()
