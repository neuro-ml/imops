import numpy as np

from .pad import pad
from .utils import AxesLike, AxesParams, broadcast_axis, fill_by_indices


def crop_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None, ratio: AxesParams = 0.5) -> np.ndarray:
    """
    Crop `x` to match `shape` along `axis`.

    Parameters
    ----------
    x
    shape
        final shape.
    axis
        axis along which `x` will be padded.
    ratio
        the fraction of the crop that will be applied to the left, `1 - ratio` will be applied to the right.
        By default `ratio=0.5`, i.e. it is applied uniformly to the left and right.
    """
    x = np.asarray(x)
    axis, shape, ratio = broadcast_axis(axis, x.ndim, shape, ratio)

    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axis))
    if (old_shape < new_shape).any():
        raise ValueError(f'The resulting shape cannot be greater than the original one: {old_shape} vs {new_shape}.')

    ndim = len(x.shape)
    ratio = fill_by_indices(np.zeros(ndim), ratio, axis)
    start = ((old_shape - new_shape) * ratio).astype(int)

    return x[tuple(map(slice, start, start + new_shape))]


def crop_to_box(x: np.ndarray, box: np.ndarray, axis: AxesLike = None, padding_values: AxesParams = None) -> np.ndarray:
    """
    Crop `x` according to `box` along `axis`.
    """
    x = np.asarray(x)
    start, stop = box
    axis, start, stop = broadcast_axis(axis, x.ndim, start, stop)

    slice_start = np.maximum(start, 0)
    slice_stop = np.minimum(stop, np.array(x.shape)[list(axis)])
    padding = np.array([slice_start - start, stop - slice_stop], dtype=int).T
    if padding_values is None and padding.any():
        raise ValueError(f"The box {box} exceeds the input's limits {x.shape}.")

    slice_start = fill_by_indices(np.zeros(x.ndim, int), slice_start, axis)
    slice_stop = fill_by_indices(x.shape, slice_stop, axis)
    x = x[tuple(map(slice, slice_start, slice_stop))]

    if padding_values is not None and padding.any():
        x = pad(x, padding, axis, padding_values)

    return x
