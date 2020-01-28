"""Model evaluation metrics."""


from typing import Union

import numpy as np
import torch


def corr_coef(
    a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]
) -> float:
    """Calculates the Pearson correlation coefficient between the inputs.

    Parameters
    ----------
    a
        First input.
    b
        Second input.

    Returns
    -------
    float
        Pearson correlation coefficient between the inputs.

    """
    if a is None or b is None:
        return None
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()
    assert a.shape == b.shape, "Inputs must be same shape"
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    cc = np.mean((a - mean_a) * (b - mean_b)) / (std_a * std_b)
    return cc


def corr_coef_chan0(
    a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]
) -> float:
    """Calculates the Pearson correlation coefficient between channel 0 of the
    inputs.

    Assumes the first dimension of the inputs is the channel dimension.

    Parameters
    ----------
    a
        First input.
    b
        Second input.

    Returns
    -------
    float
        Pearson correlation coefficient between channel 0 of the inputs.

    """
    if a is None or b is None:
        return None
    a = a[0:1,]
    b = b[0:1,]
    return corr_coef(a, b)
