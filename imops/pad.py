from typing import Callable, Sequence, Union

import numpy as np

from .backend import BackendLike
from .numeric import _NUMERIC_DEFAULT_NUM_THREADS, copy
from .utils import AxesLike, AxesParams, axis_from_dim, broadcast_axis, broadcast_to_axis, fill_by_indices


def pad(
    x: np.ndarray,
    padding: Union[AxesLike, Sequence[Sequence[int]]],
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Pad `x` according to `padding` along the `axis`.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array to pad
    padding: Union[AxesLike, Sequence[Sequence[int]]]
        if 2D array [[start_1, stop_1], ..., [start_n, stop_n]] - specifies individual padding
        for each axis from `axis`. The length of the array must either be equal to 1 or match the length of `axis`.
        If 1D array [val_1, ..., val_n] - same as [[val_1, val_1], ..., [val_n, val_n]].
        If scalar (val) - same as [[val, val]]
    axis: AxesLike
        axis along which `x` will be padded
    padding_values: Union[AxesParams, Callable]
        values to pad with, must be broadcastable to the resulting array.
        If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    padded: np.ndarray
        padded array

    Examples
    --------
    >>> padded = pad(x, 2)  # pad 2 zeros on each side of each axes
    >>> padded = pad(x, [1, 1], axis=(-1, -2))  # pad 1 zero on each side of last 2 axes
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
    new_x = copy(np.broadcast_to(new_x, new_shape), order='C', num_threads=num_threads, backend=backend)

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    # TODO: how to parallelize this?
    new_x[tuple(map(slice, start, end))] = x

    return new_x


def pad_to_shape(
    x: np.ndarray,
    shape: AxesLike,
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
    ratio: AxesParams = 0.5,
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Pad `x` to match `shape` along the `axis`.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array to pad
    shape: AxesLike
        final shape
    axis: AxesLike
        axis along which `x` will be padded
    padding_values: Union[AxesParams, Callable]
        values to pad with, must be broadcastable to the resulting array.
        If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used
    ratio: AxesParams
        float or sequence of floats describing what proportion of padding to apply on the left sides of padding axes.
        Remaining ratio of padding will be applied on the right sides
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    padded: np.ndarray
        padded array

    Examples
    --------
    >>> padded = pad_to_shape(x, [4, 5, 6])  # pad 3d array
    >>> padded = pad_to_shape(x, [4, 5], axis=[0, 1], ratio=0)  # pad first 2 axes on the right
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

    return pad(x, padding, axis, padding_values=padding_values, num_threads=num_threads, backend=backend)


def pad_to_divisible(
    x: np.ndarray,
    divisor: AxesLike,
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
    ratio: AxesParams = 0.5,
    remainder: AxesLike = 0,
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Pad `x` to be divisible by `divisor` along the `axis`.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array to pad
    divisor: AxesLike
        float or sequence of floats an incoming array shape will be divisible by
    axis: AxesLike
        axis along which the array will be padded. If None - the last `len(divisor)` axes are used
    padding_values: Union[AxesParams, Callable]
        values to pad with. If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used
    ratio: AxesParams
        float or sequence of floats describing what proportion of padding to apply on the left sides of padding axes.
        Remaining ratio of padding will be applied on the right sides
    remainder: AxesLike
        `x` will be padded such that its shape gives the remainder `remainder` when divided by `divisor`
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    padded: np.ndarray
        padded array

    Examples
    --------
    >>> x  # array of shape [2, 3, 4]
    >>> padded = pad_to_divisible(x, 6)  # pad to shape [6, 6, 6]
    >>> padded = pad_to_divisible(x, [4, 3], axis=[0, 1], ratio=1)  # pad first 2 axes on the left, shape - [4, 3, 4]
    >>> padded = pad_to_divisible(x, 3, remainder=1)  # pad to shape [4, 4, 4]
    """
    x = np.asarray(x)
    axis = axis_from_dim(axis, x.ndim)
    divisor, remainder, ratio = broadcast_to_axis(axis, divisor, remainder, ratio)

    assert np.all(remainder >= 0)
    shape = np.maximum(np.array(x.shape)[list(axis)], remainder)

    return pad_to_shape(
        x, shape + (remainder - shape) % divisor, axis, padding_values, ratio, num_threads=num_threads, backend=backend
    )


def restore_crop(
    x: np.ndarray,
    box: np.ndarray,
    shape: AxesLike,
    padding_values: Union[AxesParams, Callable] = 0,
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Pad `x` to match `shape`. The left padding is taken equal to `box`'s start.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array to pad
    box: np.ndarray
        array of shape (2, x.ndim) describing crop boundaries
    shape: AxesLike
        shape to restore crop to
    padding_values: Union[AxesParams, Callable]
        values to pad with. If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    padded: np.ndarray
        padded array

    Examples
    --------
    >>> x  # array of shape [2, 3, 4]
    >>> padded = restore_crop(x, np.array([[0, 0, 0], [2, 3, 4]]), [4, 4, 4])  # pad to shape [4, 4, 4]
    >>> padded = restore_crop(x, np.array([[0, 0, 0], [1, 1, 1]]), [4, 4, 4])  # fail, box is inconsistent with an array
    >>> padded = restore_crop(x, np.array([[1, 2, 3], [3, 5, 7]]), [3, 5, 7])  # pad to shape [3, 5, 7]
    """
    start, stop = np.asarray(box)

    assert len(shape) == x.ndim
    assert len(start) == len(stop) == x.ndim

    x = np.asarray(x)

    if (stop > shape).any() or (stop - start != x.shape).any():
        raise ValueError(
            f'The input array (of shape {x.shape}) was not obtained by cropping a '
            f'box {start, stop} from the shape {shape}.'
        )

    padding = np.array([start, shape - stop], dtype=int).T
    x = pad(x, padding, padding_values=padding_values, num_threads=num_threads, backend=backend)
    assert all(np.array(x.shape) == shape)

    return x
