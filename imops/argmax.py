from warnings import warn

import numpy as np

from .backend import Cython
from .compat import normalize_axis_index
from .src._argmax import _argmax
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
        C-contiguous result of argmax

    Examples
    --------
    ```python
    result = argmax(x, axis=-1)
    ```
    """
    if array.shape[axis] > 256:
        warn(
            "Fast argmax is only supported for array.shape[axis] <= 256. Falling back to numpy's implementation.",
            stacklevel=3,
        )

        return np.argmax(array, axis=axis)
    elif array.dtype not in (np.float32, np.float64, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32):
        warn(
            "Fast argmax is only supported for float32, float64, int16, int32, int64, uint8, uint16, uint32"
            "Falling back to numpy's implementation.",
            stacklevel=3
        )

        return np.argmax(array, axis=axis)

    # TODO: handle this case via permutations + implement the cython src functions with output arg
    if not array.data.c_contiguous:
        warn('Input array is not C-contiguous, performance can drop a lot.', stacklevel=3)

    ndim = array.ndim
    shape = array.shape
    axis = normalize_axis_index(axis, ndim)
    num_threads = normalize_num_threads(num_threads, Cython())
    num_threads = min(num_threads, 32)  # don't spawn more than 32 threads or slowdown can be occured

    pre_shape = shape[:axis]
    post_shape = shape[axis + 1 :]

    argmax_dim = shape[axis]
    pre_dim = np.prod(pre_shape) if len(pre_shape) else 1
    post_dim = np.prod(post_shape) if len(post_shape) else 1

    array = array.reshape(pre_dim, argmax_dim, post_dim)

    out = _argmax(array, pre_dim, argmax_dim, post_dim, num_threads)

    out = out.reshape(pre_shape + post_shape)

    return out
