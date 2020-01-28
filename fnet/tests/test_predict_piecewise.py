from fnet.predict_piecewise import predict_piecewise
import numpy as np
import numpy.testing as npt
import torch


class FakePredictor:
    def predict(self, x, tta=False):
        y_hat = x.copy() + 0.42
        return torch.tensor(y_hat)


def test_predict_piecewise():
    # Create pretty gradient image as test input
    shape = (1, 32, 512, 256)
    ar_in = 1
    for idx in range(1, len(shape)):
        slices = [None] * len(shape)
        slices[idx] = slice(None)
        ar_in = ar_in * np.linspace(0, 1, num=shape[idx], endpoint=False)[tuple(slices)]
    ar_in = torch.tensor(ar_in.astype(np.float32))
    predictor = FakePredictor()
    ar_out = predict_piecewise(
        predictor, ar_in, dims_max=[None, 32, 128, 64], overlaps=16
    )
    got = ar_out.numpy()
    expected = ar_in.numpy() + 0.42
    npt.assert_almost_equal(got, expected)


if __name__ == "__main__":
    test_predict_piecewise()
