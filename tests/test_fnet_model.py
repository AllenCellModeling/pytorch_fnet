import unittest
import torch
import fnet.fnet_model
import os
import pdb

def get_types(x):
    types = set()
    if isinstance(x, dict):
        for key, val in x.items():
            types.update(get_types(val))
    else:
        types.add(type(x))
    return types

class TestFnetModel(unittest.TestCase):
    def setUp(self):
        self.model = fnet.fnet_model.Model(
            nn_module = 'fnet_nn_3d',
            gpu_ids = 3,
        )
        self.x_batch = torch.rand(12, 1, 32, 16, 16)
        self.y_batch = 2*self.x_batch

    def test_move(self):
        move_start = lambda : None
        move_cpu = lambda : self.model.to_gpu(-1)
        move_1 = lambda : self.model.to_gpu(1)
        move_023 = lambda : self.model.to_gpu([0, 2, 3])
        moves = [move_start, move_cpu, move_1, move_023]

        for move in moves:
            move()
            print(self.model.gpu_ids)
            loss = self.model.do_train_iter(self.x_batch, self.y_batch)
            types_nn = get_types(self.model.net.state_dict())
            types_optim = get_types(self.model.optimizer.state_dict())
            y_pred_batch = self.model.predict(self.x_batch)
            if move == move_cpu:
                self.assertFalse(torch.cuda.FloatTensor in types_nn)
                self.assertFalse(torch.cuda.FloatTensor in types_optim)
                self.assertTrue(torch.FloatTensor in types_nn)
                self.assertTrue(torch.FloatTensor in types_optim)
            else:
                self.assertTrue(torch.cuda.FloatTensor in types_nn)
                self.assertTrue(torch.cuda.FloatTensor in types_optim)
                self.assertFalse(torch.FloatTensor in types_nn)
                self.assertFalse(torch.FloatTensor in types_optim)

    def test_save_load(self):
        path_save = 'tests/tmp/model.p'
        self.model.to_gpu(1)
        loss = self.model.do_train_iter(self.x_batch, self.y_batch)
        self.model.save_state(path_save)
        loss = self.model.do_train_iter(self.x_batch, self.y_batch)

        # loaded state should not be on gpu
        state_model = torch.load(path_save)
        types = get_types(state_model)
        self.assertFalse(torch.cuda.FloatTensor in types)
        self.assertTrue(torch.FloatTensor in types)

        path_save_dir = os.path.dirname(path_save)
        new_model = fnet.load_model_from_dir(path_save_dir, gpu_ids=3)
        types_loaded = get_types(new_model.get_state())
        loss = new_model.do_train_iter(self.x_batch, self.y_batch)

if __name__ == '__main__':
    unittest.main()
