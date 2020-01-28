from scipy.signal import triang
from typing import Union, List
import numpy as np
import torch


def _get_weights(shape):
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        weights = weights * triang(size)[tuple(slicey)]
    return np.broadcast_to(weights, shape_in).astype(np.float32)


def _predict_piecewise_recurse(
    predictor,
    ar_in: np.ndarray,
    dims_max: Union[int, List[int]],
    overlaps: Union[int, List[int]],
    **predict_kwargs,
):
    """Performs piecewise prediction recursively."""
    if tuple(ar_in.shape[1:]) == tuple(dims_max[1:]):
        ar_out = predictor.predict(ar_in, **predict_kwargs).numpy().astype(np.float32)
        ar_weight = _get_weights(ar_out.shape)
        return ar_out * ar_weight, ar_weight
    dim = None
    # Find first dim where input > max
    for idx_d in range(1, ar_in.ndim):
        if ar_in.shape[idx_d] > dims_max[idx_d]:
            dim = idx_d
            break
    # Size of channel dim is unknown until after first prediction
    shape_out = [None] + list(ar_in.shape[1:])
    ar_out = None
    ar_weight = None
    offset = 0
    done = False
    while not done:
        slices = [slice(None)] * ar_in.ndim
        end = offset + dims_max[dim]
        slices[dim] = slice(offset, end)
        slices = tuple(slices)
        ar_in_sub = ar_in[slices]
        pred_sub, pred_weight_sub = _predict_piecewise_recurse(
            predictor, ar_in_sub, dims_max, overlaps, **predict_kwargs
        )
        if ar_out is None or ar_weight is None:
            shape_out[0] = pred_sub.shape[0]  # Set channel dim for output
            ar_out = np.zeros(shape_out, dtype=pred_sub.dtype)
            ar_weight = np.zeros(shape_out, dtype=pred_weight_sub.dtype)
        ar_out[slices] += pred_sub
        ar_weight[slices] += pred_weight_sub
        offset += dims_max[dim] - overlaps[dim]
        if end == ar_in.shape[dim]:
            done = True
        elif offset + dims_max[dim] > ar_in.shape[dim]:
            offset = ar_in.shape[dim] - dims_max[dim]
    return ar_out, ar_weight


def predict_piecewise(
    predictor,
    tensor_in: torch.Tensor,
    dims_max: Union[int, List[int]] = 64,
    overlaps: Union[int, List[int]] = 0,
    **predict_kwargs,
) -> torch.Tensor:
    """Performs piecewise prediction and combines results.

    Parameters
    ----------
    predictor
         An object with a predict() method.
    tensor_in
         Tensor to be input into predictor piecewise. Should be 3d or 4d with
         with the first dimension channel.
    dims_max
         Specifies dimensions of each sub prediction.
    overlaps
         Specifies overlap along each dimension for sub predictions.
    **predict_kwargs
        Kwargs to pass to predict method.

    Returns
    -------
    torch.Tensor
         Prediction with size tensor_in.size().

    """
    assert isinstance(tensor_in, torch.Tensor)
    assert len(tensor_in.size()) > 2
    shape_in = tuple(tensor_in.size())
    n_dim = len(shape_in)
    if isinstance(dims_max, int):
        dims_max = [dims_max] * n_dim
    for idx_d in range(1, n_dim):
        if dims_max[idx_d] > shape_in[idx_d]:
            dims_max[idx_d] = shape_in[idx_d]
    if isinstance(overlaps, int):
        overlaps = [overlaps] * n_dim
    assert len(dims_max) == len(overlaps) == n_dim
    # Remove restrictions on channel dimension.
    dims_max[0] = None
    overlaps[0] = None
    ar_in = tensor_in.numpy()
    ar_out, ar_weight = _predict_piecewise_recurse(
        predictor, ar_in, dims_max=dims_max, overlaps=overlaps, **predict_kwargs
    )
    # tifffile.imsave('debug/ar_sum.tif', ar_out)
    mask = ar_weight > 0.0
    ar_out[mask] = ar_out[mask] / ar_weight[mask]
    # tifffile.imsave('debug/ar_weight.tif', ar_weight)
    # tifffile.imsave('debug/ar_out.tif', ar_out)
    return torch.tensor(ar_out)
