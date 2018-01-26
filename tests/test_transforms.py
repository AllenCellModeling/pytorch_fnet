import unittest
import numpy as np
import fnet.transforms

class TestTransforms(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(linewidth=120)
        self.rng = np.random.RandomState(42)

    def test_padder(self):
        shape = (11, 64, 88)
        # shape = (6, 25)
        img = self.rng.randint(100, size=(shape))

        padders = [
            fnet.transforms.Padder('+', mode='reflect'),
            fnet.transforms.Padder((3, 2, '+')),
        ]
        shapes_exp = [
            tuple(np.ceil(np.array(shape)/16).astype(np.int)*16),
            (17, 68, 96),
        ]
        for idx, padder in enumerate(padders):
            img_padded = padder(img)
            self.assertEqual(shapes_exp[idx], img_padded.shape)

            img_undo = padder.undo_last(img_padded)
            self.assertTrue(np.array_equal(img, img_undo))

