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

    def test_cropper(self):
        shape = (20, 32, 63)
        img = self.rng.randint(100, size=(shape))
        transformers = [
            fnet.transforms.Cropper('-'),
            fnet.transforms.Cropper(cropping=(3,4,5), offset=(1,2,3)),
        ]
        shapes_exp = [
            (16, 32, 48),
            (17, 28, 58),
        ]
        for idx, transformer in enumerate(transformers):
            img_trans = transformer(img)
            # self.assertEqual(shapes_exp[idx], img_trans.shape)
            img_undo = transformer.undo_last(img_trans)
        
    def test_propper(self):
        print('testing propper')
        shape = (18, 55)
        img = self.rng.randint(100, size=(shape))

        proppers = [
            fnet.transforms.Propper('+', mode='reflect'),
            fnet.transforms.Propper('-'),
        ]
        shapes_exp = [
            (32, 64),
            (16, 48),
        ]
        for idx, propper in enumerate(proppers):
            img_propped = propper(img)
            self.assertEqual(shapes_exp[idx], img_propped.shape)
            img_undo = propper.undo_last(img_propped)
            if propper.action != '-':
                self.assertTrue(np.array_equal(img, img_undo))
        
