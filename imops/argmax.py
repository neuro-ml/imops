from warnings import warn

import numpy as np

from .backend import Cython
from .compat import normalize_axis_index
from .src._argmax import _inner_argmax, _inner_argmax_out, _outer_argmax, _outer_inner_argmax
from .utils import AxesLike, normalize_num_threads


def argmax(array: np.ndarray, axis: AxesLike, num_threads: int = -1):
    """
    Fast parallel implementation of argmax

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array
    axis: AxesLike
        axis along which argmax is applied
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used

    Returns
    -------
    out: np.ndarray
        result of argmax

    Examples
    --------
    ```python
    result = argmax(x, axis=-1)
    ```
    """
    ndim = array.ndim
    shape = array.shape
    axis = normalize_axis_index(axis, ndim)
    num_threads = normalize_num_threads(num_threads, Cython())

    # don't spawn more than 32 threads or slowdown can be occured
    num_threads = min(num_threads, 32)

    if shape[axis] > 256:
        warn(
            "Fast argmax is only supported for array.shape[axis] <= 256. Falling back to numpy's implementation.",
            stacklevel=3,
        )

        return np.argmax(array, axis=axis)

    pre_shape = shape[:axis]
    post_shape = shape[axis + 1 :]

    argmax_dim = shape[axis]
    pre_dim = np.prod(pre_shape) if len(pre_shape) else 1
    post_dim = np.prod(post_shape) if len(post_shape) else 1

    # Use simplier implementations if possible
    if axis == ndim - 1:
        array = array.reshape(pre_dim, argmax_dim)
        out = _outer_argmax(array, argmax_dim, pre_dim, num_threads)

    elif axis == 0:
        array = array.reshape(argmax_dim, post_dim)
        out = _inner_argmax(array, argmax_dim, post_dim, num_threads)

    # Don't use super-parallel implementation if possible
    elif pre_dim < num_threads:
        array = array.reshape(pre_dim, argmax_dim, post_dim)
        out = np.empty((pre_dim, post_dim), dtype=np.uint8)

        for array_part, out_part in zip(array, out):
            _inner_argmax_out(array_part, out_part, argmax_dim, post_dim, num_threads)

    else:
        array = array.reshape(pre_dim, argmax_dim, post_dim)
        out = _outer_inner_argmax(array, pre_dim, argmax_dim, post_dim, num_threads)

    out = out.reshape(pre_shape + post_shape)

    return out
