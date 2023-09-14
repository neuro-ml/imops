import numpy as np

from .backend import BackendLike
from .numeric import _NUMERIC_DEFAULT_NUM_THREADS
from .pad import pad
from .utils import AxesLike, AxesParams, broadcast_axis, fill_by_indices


def crop_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None, ratio: AxesParams = 0.5) -> np.ndarray:
    """
    Crop `x` to match `shape` along `axis`.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array
    shape: AxesLike
        final shape
    axis: AxesLike
        axis along which `x` will be padded
    ratio: AxesParams
        float or sequence of floats describing what proportion of cropping to apply on the left sides of cropping axes.
        Remaining ratio of cropping will be applied on the right sides

    Returns
    -------
    cropped: np.ndarray
        cropped array

    Examples
    --------
    >>> x  # array of shape [2, 3, 4]
    >>> cropped = crop_to_shape(x, [1, 2, 3], ratio=0)  # crop to shape [1, 2, 3] from the right
    >>> cropped = crop_to_shape(x, 2, axis=1, ratio=1)  # crop to shape [2, 2, 4] from the left
    >>> cropped = crop_to_shape(x, [3, 4, 5])  # fail due to bigger resulting shape
    """
    x = np.asarray(x)
    axis, shape, ratio = broadcast_axis(axis, x.ndim, shape, ratio)

    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axis))
    if (old_shape < new_shape).any():
        raise ValueError(f'The resulting shape cannot be greater than the original one: {old_shape} vs {new_shape}.')

    ndim = len(x.shape)
    ratio = fill_by_indices(np.zeros(ndim), ratio, axis)
    start = ((old_shape - new_shape) * ratio).astype(int)

    # TODO: Create contiguous array?
    return x[tuple(map(slice, start, start + new_shape))]


def crop_to_box(
    x: np.ndarray,
    box: np.ndarray,
    axis: AxesLike = None,
    padding_values: AxesParams = None,
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Crop `x` according to `box` along `axis`.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array
    box: np.ndarray
        array of shape (2, x.ndim or len(axis) if axis is passed) describing crop boundaries
    axis: AxesLike
        axis along which `x` will be cropped
    padding_values: AxesParams
        values to pad with if box exceeds the input's limits
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    cropped: np.ndarray
        cropped array

    Examples
    --------
    >>> x  # array of shape [2, 3, 4]
    >>> cropped = crop_to_box(x, np.array([[0, 0, 0], [1, 1, 1]]))  # crop to shape [1, 1, 1]
    >>> cropped = crop_to_box(x, np.array([[0, 0, 0], [5, 5, 5]]))  # fail, box exceeds the input's limits
    >>> cropped = crop_to_box(x, np.array([[0], [5]]), axis=0, padding_values=0)  # pad with 0-s to shape [5, 3, 4]
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
    # TODO: Create contiguous array?
    x = x[tuple(map(slice, slice_start, slice_stop))]

    if padding_values is not None and padding.any():
        x = pad(x, padding, axis, padding_values, num_threads=num_threads, backend=backend)

    return x
