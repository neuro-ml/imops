import numpy as np
from .backend import BackendLike, resolve_backend
from .compat import normalize_axis_index
from .utils import normalize_num_threads, AxesLike

from .src._argmax import _inner_argmax, _outer_argmax, _outer_inner_argmax


def argmax(array: np.ndarray, axis: AxesLike, num_threads: int = -1, backend: BackendLike = None):
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
    backend: BackendLike
        which backend to use. `numba`, `cython` and `scipy` are available, `cython` is used by default

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
    backend = resolve_backend(backend, warn_stacklevel=4)
    if backend.name not in ('Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    ndim = array.ndim
    shape = array.shape
    axis = normalize_axis_index(axis, ndim)
    num_threads = normalize_num_threads(num_threads)

    assert shape[axis] <= 256

    pre_shape = shape[:axis]
    post_shape = shape[axis + 1 :]

    argmax_dim = shape[axis]
    pre_dim = np.prod(pre_shape) if len(pre_shape) else 1
    post_dim = np.prod(post_shape) if len(post_shape) else 1

    if axis == ndim - 1:
        array = array.reshape(pre_dim, argmax_dim)
        out = _outer_argmax(array, argmax_dim, pre_dim, num_threads)

    elif axis == 0:
        array = array.reshape(argmax_dim, post_dim)
        out = _inner_argmax(array, argmax_dim, post_dim, num_threads)

    else:
        array = array.reshape(pre_dim, argmax_dim, post_dim)
        out = _outer_inner_argmax(array, pre_dim, argmax_dim, post_dim, num_threads)

    out = out.reshape(pre_shape + post_shape)

    return out
