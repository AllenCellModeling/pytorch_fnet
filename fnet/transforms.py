from typing import Optional
import logging

import numpy as np
import scipy


logger = logging.getLogger(__name__)


class Normalize:
    def __init__(self, per_dim=None):
        """Class version of normalize function."""
        self.per_dim = per_dim

    def __call__(self, x):
        return normalize(x, per_dim=self.per_dim)

    def __repr__(self):
        return "Normalize({})".format(self.per_dim)


class ToFloat:
    def __call__(self, x):
        return x.astype(np.float32)

    def __repr__(self):
        return "ToFloat()"


def normalize(img, per_dim=None):
    """Subtract mean, set STD to 1.0

    Parameters:
      per_dim: normalize along other axes dimensions not equal to per dim
    """
    axis = tuple([i for i in range(img.ndim) if i != per_dim])
    slices = tuple(
        [slice(None) if i == per_dim else np.newaxis for i in range(img.ndim)]
    )  # to handle broadcasting
    result = img.astype(np.float32)
    result -= np.mean(result, axis=axis)[slices]
    result /= np.std(result, axis=axis)[slices]
    return result


def do_nothing(img):
    return img.astype(np.float)


class Propper:
    """Padder + Cropper"""

    def __init__(self, action="-", **kwargs):
        self.action = action
        if self.action in ["+", "pad"]:
            self.transformer = Padder(**kwargs)
        elif self.action in ["-", "crop"]:
            self.transformer = Cropper(**kwargs)
        else:
            raise NotImplementedError

    def __repr__(self):
        return repr(self.transformer)

    def __call__(self, x_in):
        return self.transformer(x_in)

    def undo_last(self, x_in):
        return self.transformer.undo_last(x_in)


class Padder(object):
    def __init__(self, padding="+", by=16, mode="constant"):
        """
        padding: '+', int, sequence
          '+': pad dimensions up to multiple of "by"
          int: pad each dimension by this value
          sequence: pad each dimensions by corresponding value in sequence
        by: int
          for use with '+' padding option
        mode: str
          passed to numpy.pad function
        """
        self.padding = padding
        self.by = by
        self.mode = mode
        self.pads = {}
        self.last_pad = None

    def __repr__(self):
        return "Padder{}".format((self.padding, self.by, self.mode))

    def _calc_pad_width(self, shape_in):
        if isinstance(self.padding, (str, int)):
            paddings = (self.padding,) * len(shape_in)
        else:
            paddings = self.padding
        pad_width = []
        for i in range(len(shape_in)):
            if isinstance(paddings[i], int):
                pad_width.append((paddings[i],) * 2)
            elif paddings[i] == "+":
                padding_total = (
                    int(np.ceil(1.0 * shape_in[i] / self.by) * self.by) - shape_in[i]
                )
                pad_left = padding_total // 2
                pad_right = padding_total - pad_left
                pad_width.append((pad_left, pad_right))
        assert len(pad_width) == len(shape_in)
        return pad_width

    def undo_last(self, x_in):
        """Crops input so its dimensions matches dimensions of last input to __call__."""
        assert x_in.shape == self.last_pad["shape_out"]
        slices = [
            slice(a, -b) if (a, b) != (0, 0) else slice(None)
            for a, b in self.last_pad["pad_width"]
        ]
        return x_in[slices].copy()

    def __call__(self, x_in):
        shape_in = x_in.shape
        pad_width = self.pads.get(shape_in, self._calc_pad_width(shape_in))
        x_out = np.pad(x_in, pad_width, mode=self.mode)
        if shape_in not in self.pads:
            self.pads[shape_in] = pad_width
        self.last_pad = {
            "shape_in": shape_in,
            "pad_width": pad_width,
            "shape_out": x_out.shape,
        }
        return x_out


class Cropper(object):
    def __init__(
        self, cropping="-", by=16, offset="mid", n_max_pixels=9732096, dims_no_crop=None
    ):
        """Crop input array to given shape."""
        self.cropping = cropping
        self.offset = offset
        self.by = by
        self.n_max_pixels = n_max_pixels
        self.dims_no_crop = (
            [dims_no_crop] if isinstance(dims_no_crop, int) else dims_no_crop
        )
        self.crops = {}
        self.last_crop = None

    def __repr__(self):
        return "Cropper{}".format(
            (self.cropping, self.by, self.offset, self.n_max_pixels, self.dims_no_crop)
        )

    def _adjust_shape_crop(self, shape_crop):
        shape_crop_new = list(shape_crop)
        prod_shape = np.prod(shape_crop_new)
        idx_dim_reduce = 0
        order_dim_reduce = list(
            range(len(shape_crop))[-2:]
        )  # alternate between last two dimensions
        while prod_shape > self.n_max_pixels:
            dim = order_dim_reduce[idx_dim_reduce]
            if not (dim == 0 and shape_crop_new[dim] <= 64):
                shape_crop_new[dim] -= self.by
                prod_shape = np.prod(shape_crop_new)
            idx_dim_reduce += 1
            if idx_dim_reduce >= len(order_dim_reduce):
                idx_dim_reduce = 0
        value = tuple(shape_crop_new)
        return value

    def _calc_shape_crop(self, shape_in):
        croppings = (
            (self.cropping,) * len(shape_in)
            if isinstance(self.cropping, (str, int))
            else self.cropping
        )
        shape_crop = []
        for i in range(len(shape_in)):
            if (croppings[i] is None) or (
                self.dims_no_crop is not None and i in self.dims_no_crop
            ):
                shape_crop.append(shape_in[i])
            elif isinstance(croppings[i], int):
                shape_crop.append(shape_in[i] - croppings[i])
            elif croppings[i] == "-":
                shape_crop.append(shape_in[i] // self.by * self.by)
            else:
                raise NotImplementedError
        if self.n_max_pixels is not None:
            shape_crop = self._adjust_shape_crop(shape_crop)
        self.crops[shape_in]["shape_crop"] = shape_crop
        return shape_crop

    def _calc_offsets_crop(self, shape_in, shape_crop):
        offsets = (
            (self.offset,) * len(shape_in)
            if isinstance(self.offset, (str, int))
            else self.offset
        )
        offsets_crop = []
        for i in range(len(shape_in)):
            offset = (
                (shape_in[i] - shape_crop[i]) // 2
                if offsets[i] == "mid"
                else offsets[i]
            )
            if offset + shape_crop[i] > shape_in[i]:
                logger.error(
                    f"Cannot crop outsize image dimensions ({offset}:{offset + shape_crop[i]} for dim {i})"
                )
                raise AttributeError
            offsets_crop.append(offset)
        self.crops[shape_in]["offsets_crop"] = offsets_crop
        return offsets_crop

    def _calc_slices(self, shape_in):
        shape_crop = self._calc_shape_crop(shape_in)
        offsets_crop = self._calc_offsets_crop(shape_in, shape_crop)
        slices = [
            slice(offsets_crop[i], offsets_crop[i] + shape_crop[i])
            for i in range(len(shape_in))
        ]
        self.crops[shape_in]["slices"] = slices
        return slices

    def __call__(self, x_in):
        shape_in = x_in.shape
        if shape_in in self.crops:
            slices = self.crops[shape_in]["slices"]
        else:
            self.crops[shape_in] = {}
            slices = self._calc_slices(shape_in)
        x_out = x_in[slices].copy()
        self.last_crop = {
            "shape_in": shape_in,
            "slices": slices,
            "shape_out": x_out.shape,
        }
        return x_out

    def undo_last(self, x_in):
        """Pads input with zeros so its dimensions matches dimensions of last input to __call__."""
        assert x_in.shape == self.last_crop["shape_out"]
        shape_out = self.last_crop["shape_in"]
        slices = self.last_crop["slices"]
        x_out = np.zeros(shape_out, dtype=x_in.dtype)
        x_out[slices] = x_in
        return x_out


class Resizer(object):
    def __init__(self, factors, per_dim=None):
        """
        Parameters:
          factors: tuple of resizing factors for each dimension of the input array
          per_dim: normalize along other axes dimensions not equal to per dim
        """
        self.factors = factors
        self.per_dim = per_dim

    def __call__(self, x):
        if self.per_dim is None:
            return scipy.ndimage.zoom(x, (self.factors), mode="nearest")
        ars_resized = list()
        for idx in range(x.shape[self.per_dim]):
            slices = tuple(
                [idx if i == self.per_dim else slice(None) for i in range(x.ndim)]
            )
            ars_resized.append(
                scipy.ndimage.zoom(x[slices], self.factors, mode="nearest")
            )
        return np.stack(ars_resized, axis=self.per_dim)

    def __repr__(self):
        return "Resizer({:s}, {})".format(str(self.factors), self.per_dim)


class Capper(object):
    def __init__(self, low=None, hi=None):
        self._low = low
        self._hi = hi

    def __call__(self, ar):
        result = ar.copy()
        if self._hi is not None:
            result[result > self._hi] = self._hi
        if self._low is not None:
            result[result < self._low] = self._low
        return result

    def __repr__(self):
        return "Capper({}, {})".format(self._low, self._hi)


def flip_y(ar: np.ndarray) -> np.ndarray:
    """Flip array along y axis.

    Array dimensions should end in YX.

    Parameters
    ----------
    ar
        Input array to be flipped.

    Returns
    -------
    np.ndarray
        Flipped array.

    """
    return np.flip(ar, axis=-2)


def flip_x(ar: np.ndarray) -> np.ndarray:
    """Flip array along x axis.

    Array dimensions should end in YX.

    Parameters
    ----------
    ar
        Input array to be flipped.

    Returns
    -------
    np.ndarray
        Flipped array.

    """
    return np.flip(ar, axis=-1)


def norm_around_center(ar: np.ndarray, z_center: Optional[int] = None):
    """Returns normalized version of input array.

    The array will be normalized with respect to the mean, std pixel intensity
    of the sub-array of length 32 in the z-dimension centered around the
    array's "z_center".

    Parameters
    ----------
    ar
        Input 3d array to be normalized.
    z_center
        Z-index of cell centers.

    Returns
    -------
    np.ndarray
       Nomralized array, dtype = float32

    """
    if ar.ndim != 3:
        raise ValueError("Input array must be 3d")
    if ar.shape[0] < 32:
        raise ValueError("Input array must be at least length 32 in first dimension")
    if z_center is None:
        z_center = ar.shape[0] // 2
    chunk_zlen = 32
    z_start = z_center - chunk_zlen // 2
    if z_start < 0:
        z_start = 0
        logger.warn(f"Warning: z_start set to {z_start}")
    if (z_start + chunk_zlen) > ar.shape[0]:
        z_start = ar.shape[0] - chunk_zlen
        logger.warn(f"Warning: z_start set to {z_start}")
    chunk = ar[z_start : z_start + chunk_zlen, :, :]
    ar = ar - chunk.mean()
    ar = ar / chunk.std()
    return ar.astype(np.float32)
