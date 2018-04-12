import fnet
import fnet.fnet_model
import os
import torch
import unittest
import pdb

def get_types(x):
    """Recursively find all types in state dictionary."""
    types = set()
    if isinstance(x, dict):
        for key, val in x.items():
            types.update(get_types(val))
    else:
        types.add(type(x))
    return types

class TestFnetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # place symlink to dummy pytorch neural net in fnet/nn_modules
        cls.dirname_test = os.path.dirname(__file__)
        path_src = os.path.join(cls.dirname_test, 'data', 'nn_test.py')
        cls.path_nn = os.path.join(cls.dirname_test, '..',
                                    'fnet', 'nn_modules', os.path.basename(path_src))
        if os.path.exists(cls.path_nn):
            os.unlink(cls.path_nn)
        os.symlink(path_src, cls.path_nn)

        # create data batch
        torch.manual_seed(42)
        x = torch.rand(16)
        cls.batch_x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)  # size: (1, 1, 16)
        cls.batch_y = cls.batch_x*4.2

        # create list of gpu combinations to test
        cls.test_gpu_ids = list()
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
            cls.test_gpu_ids.extend(devices)
            cls.test_gpu_ids.append(devices)  # all devices simultaneously
        cls.test_gpu_ids.append(-1)  # CPU
        print('gpu_ids to be tested:', cls.test_gpu_ids)

    @classmethod
    def tearDownClass(cls):
        # remove symlink placed in fnet/nn_modules
        os.unlink(cls.path_nn)

    def test_train_save_load_predict(self):
        """Train a new model, save it, load it, and perform prediction."""
        # Make temp model save directory
        path_save_dir = os.path.join(os.path.dirname(__file__), '.tmp_saved_model')
        path_save_model = os.path.join(path_save_dir, 'model.p')
        if not os.path.exists(path_save_dir):
            os.makedirs(path_save_dir)
        for gpu_id in self.test_gpu_ids:
            model = fnet.fnet_model.Model(nn_module = 'nn_test', gpu_ids = gpu_id)
            for idx in range(2):
                loss = model.do_train_iter(self.batch_x, self.batch_y)
            batch_y_pred = model.predict(self.batch_x)
            model.save_state(path_save_model)
            state_loaded = torch.load(path_save_model)
            types_state_loaded = get_types(state_loaded)
            self.assertTrue(
                torch.cuda.FloatTensor not in types_state_loaded,
                msg = 'CUDA tensors should not be in saved state',
            )
            model_loaded = fnet.load_model_from_dir(path_save_dir, gpu_ids=gpu_id)
            batch_y_pred_loaded = model_loaded.predict(self.batch_x)
            self.assertTrue(torch.equal(batch_y_pred, batch_y_pred_loaded))

        # Remove saved temp model
        os.remove(path_save_model)
        os.rmdir(path_save_dir)

    def test_load_previous_fnet_model(self):
        """Load previously saved model and perform prediction."""
        model = fnet.load_model_from_dir(os.path.join(self.dirname_test, 'data', 'model_test'), gpu_ids=-1)
        batch_y_pred = model.predict(self.batch_x)

    def test_move(self):
        """Move of models to different GPUs."""
        model = fnet.fnet_model.Model(nn_module = 'nn_test', gpu_ids = -1)
        loss = model.do_train_iter(self.batch_x, self.batch_y)
        for gpu_id in self.test_gpu_ids:
            model.to_gpu(gpu_id)
            self.assertEqual(
                sorted(model.gpu_ids),
                sorted([gpu_id] if isinstance(gpu_id, int) else gpu_id),
            )
            batch_y_pred = model.predict(self.batch_x)
            types_nn = get_types(model.net.state_dict())
            types_optim = get_types(model.optimizer.state_dict())
            types_pred = get_types(batch_y_pred)
            self.assertFalse(torch.cuda.FloatTensor in types_pred)
            for types in [types_nn, types_optim]:
                if gpu_id == -1:  # CPU case
                    self.assertFalse(torch.cuda.FloatTensor in types)
                    self.assertTrue(torch.FloatTensor in types)
                else:
                    self.assertTrue(torch.cuda.FloatTensor in types)
                    self.assertFalse(torch.FloatTensor in types)
