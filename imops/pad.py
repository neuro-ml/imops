from typing import Callable, Sequence, Union

import numpy as np

from .utils import AxesLike, AxesParams, axis_from_dim, broadcast_axis, broadcast_to_axis, fill_by_indices


def pad(
    x: np.ndarray,
    padding: Union[AxesLike, Sequence[Sequence[int]]],
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
) -> np.ndarray:
    """
    Pad `x` according to `padding` along the `axes`.

    Parameters
    ----------
    x
        tensor to pad.
    padding
        if 2D array [[start_1, stop_1], ..., [start_n, stop_n]] - specifies individual padding
        for each axis from `axis`. The length of the array must either be equal to 1 or match the length of `axis`.
        If 1D array [val_1, ..., val_n] - same as [[val_1, val_1], ..., [val_n, val_n]].
        If scalar (val) - same as [[val, val]].
    padding_values
        values to pad with, must be broadcastable to the resulting array.
        If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used.
    axis
        axis along which `x` will be padded.
    """
    x = np.asarray(x)
    padding = np.asarray(padding)
    if padding.ndim < 2:
        padding = padding.reshape(-1, 1)
    axis = axis_from_dim(axis, x.ndim)
    padding = np.asarray(fill_by_indices(np.zeros((x.ndim, 2), dtype=int), np.atleast_2d(padding), axis))
    if (padding < 0).any():
        raise ValueError(f'Padding must be non-negative: {padding.tolist()}.')
    if callable(padding_values):
        padding_values = padding_values(x)

    new_shape = np.array(x.shape) + np.sum(padding, axis=1)
    new_x = np.array(padding_values, dtype=x.dtype)
    new_x = np.broadcast_to(new_x, new_shape).copy()

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    new_x[tuple(map(slice, start, end))] = x

    return new_x


def pad_to_shape(
    x: np.ndarray,
    shape: AxesLike,
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
    ratio: AxesParams = 0.5,
) -> np.ndarray:
    """
    Pad `x` to match `shape` along the `axis`.

    Parameters
    ----------
    x
    shape
        final shape.
    padding_values
        values to pad with. If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used.
    axis
        axis along which `x` will be padded.
    ratio
        the fraction of the padding that will be applied to the left, `1.0 - ratio` will be applied to the right.
        By default `ratio=0.5`, i.e. it is applied uniformly to the left and right.
    """
    x = np.asarray(x)
    axis, shape, ratio = broadcast_axis(axis, x.ndim, shape, ratio)

    old_shape = np.array(x.shape)[list(axis)]
    if (old_shape > shape).any():
        shape = fill_by_indices(x.shape, shape, axis)
        raise ValueError(f'The resulting shape cannot be smaller than the original: {x.shape} vs {shape}.')

    delta = shape - old_shape
    start = (delta * ratio).astype(int)
    padding = np.array((start, delta - start)).T.astype(int)

    return pad(x, padding, axis, padding_values=padding_values)


def pad_to_divisible(
    x: np.ndarray,
    divisor: AxesLike,
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
    ratio: AxesParams = 0.5,
    remainder: AxesLike = 0,
):
    """
    Pad `x` to be divisible by `divisor` along the `axes`.

    Parameters
    ----------
    x
    divisor
        a value an incoming array should be divisible by.
    remainder
        `x` will be padded such that its shape gives the remainder `remainder` when divided by `divisor`.
    axis
        axes along which the array will be padded. If None - the last `len(divisor)` axes are used.
    padding_values
        values to pad with. If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used.
    ratio
        the fraction of the padding that will be applied to the left, `1 - ratio` will be applied to the right.
    References
    ----------
    `pad_to_shape`
    """
    x = np.asarray(x)
    axis = axis_from_dim(axis, x.ndim)
    divisor, remainder, ratio = broadcast_to_axis(axis, divisor, remainder, ratio)

    assert np.all(remainder >= 0)
    shape = np.maximum(np.array(x.shape)[list(axis)], remainder)

    return pad_to_shape(x, shape + (remainder - shape) % divisor, axis, padding_values, ratio)


def restore_crop(x: np.ndarray, box: np.ndarray, shape: AxesLike, padding_values: AxesParams = 0) -> np.ndarray:
    """
    Pad `x` to match `shape`. The left padding is taken equal to `box`'s start.
    """
    x = np.asarray(x)
    assert len(shape) == x.ndim
    start, stop = np.asarray(box)

    if (stop > shape).any() or (stop - start != x.shape).any():
        raise ValueError(
            f'The input array (of shape {x.shape}) was not obtained by cropping a '
            f'box {start, stop} from the shape {shape}.'
        )

    padding = np.array([start, shape - stop], dtype=int).T
    x = pad(x, padding, padding_values=padding_values)
    assert all(np.array(x.shape) == shape)

    return x
