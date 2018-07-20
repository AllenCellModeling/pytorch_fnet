import fnet
import fnet.fnet_model
import numpy as np
import os
import pdb
import torch
import unittest

def get_devices(x):
    """Recursively find all devices in state dictionary."""
    devices = set()
    if isinstance(x, dict):
        for key, val in x.items():
            devices.update(get_devices(val))
    else:
        if isinstance(x, torch.Tensor):
            if x.device.type == 'cpu':
                devices.add(-1)
            else:
                devices.add(x.device.index)
    return devices

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
            cls.test_gpu_ids.append(devices[-2:][::-1])  # last two devices, reverse order
        cls.test_gpu_ids.append(-1)  # CPU
        print('gpu_ids to be tested:', cls.test_gpu_ids)

    @classmethod
    def tearDownClass(cls):
        # remove symlink placed in fnet/nn_modules
        os.unlink(cls.path_nn)

    def test_train_save_load_predict(self):
        """Train a new model, save it, load it, and perform prediction."""
        # Make temp model save directory
        rng = np.random.RandomState(42)
        path_save_dir = os.path.join(os.path.dirname(__file__), '.tmp_saved_model')
        path_save_model = os.path.join(path_save_dir, 'model.p')
        if not os.path.exists(path_save_dir):
            os.makedirs(path_save_dir)
        for gpu_id in self.test_gpu_ids:
            test_param = rng.randint(1000)
            model = fnet.fnet_model.Model(
                nn_module = 'nn_test',
                nn_kwargs = {'test_param': test_param},
                gpu_ids = gpu_id,
            )
            for idx in range(2):
                loss = model.do_train_iter(self.batch_x, self.batch_y)
            batch_y_pred = model.predict(self.batch_x)
            model.save_state(path_save_model)
            state_loaded = torch.load(path_save_model)
            devices_state_loaded = get_devices(state_loaded)
            self.assertEqual(tuple(devices_state_loaded), (-1, ))  # saved state should only have CPU tensors
            
            model_loaded = fnet.load_model(path_save_dir, gpu_ids=gpu_id)
            batch_y_pred_loaded = model_loaded.predict(self.batch_x)
            self.assertTrue(torch.equal(batch_y_pred, batch_y_pred_loaded))
            self.assertEqual(model_loaded.net.test_param, test_param)

        # Remove saved temp model
        os.remove(path_save_model)
        os.rmdir(path_save_dir)

    def test_load_previous_fnet_model(self):
        """Load previously saved model and perform prediction."""
        model = fnet.load_model(os.path.join(self.dirname_test, 'data', 'model_test'), gpu_ids=-1)
        batch_y_pred = model.predict(self.batch_x)
        self.assertEqual(model.net.test_param, 42)

    def test_move(self):
        """Move models to different GPUs."""
        model = fnet.fnet_model.Model(nn_module = 'nn_test', gpu_ids = -1)
        loss = model.do_train_iter(self.batch_x, self.batch_y)
        for gpu_id in self.test_gpu_ids:
            device_exp = gpu_id if isinstance(gpu_id, int) else gpu_id[0]
            model.to_gpu(gpu_id)
            self.assertEqual(
                sorted(model.gpu_ids),
                sorted([gpu_id] if isinstance(gpu_id, int) else gpu_id),
            )
            batch_y_pred = model.predict(self.batch_x)
            devices_nn = get_devices(model.net.state_dict())
            devices_optim = get_devices(model.optimizer.state_dict())
            devices_pred = get_devices(batch_y_pred)
            self.assertEqual(tuple(devices_pred), (-1, ))  # predictions should CPU tensors
            for devices in [devices_nn, devices_optim]:
                self.assertEqual(tuple(devices), (device_exp, ))
