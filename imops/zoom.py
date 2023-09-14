from platform import python_version
from typing import Callable, Sequence, Union
from warnings import warn

import numpy as np
from scipy.ndimage import zoom as _scipy_zoom

from .backend import BackendLike, resolve_backend
from .src._fast_zoom import (
    _zoom3d_linear as cython_fast_zoom3d_linear,
    _zoom3d_nearest as cython_fast_zoom3d_nearest,
    _zoom4d_linear as cython_fast_zoom4d_linear,
    _zoom4d_nearest as cython_fast_zoom4d_nearest,
)
from .src._zoom import (
    _zoom3d_linear as cython_zoom3d_linear,
    _zoom3d_nearest as cython_zoom3d_nearest,
    _zoom4d_linear as cython_zoom4d_linear,
    _zoom4d_nearest as cython_zoom4d_nearest,
)
from .utils import (
    AxesLike,
    AxesParams,
    broadcast_axis,
    fill_by_indices,
    get_c_contiguous_permutaion,
    inverse_permutation,
    normalize_num_threads,
)


def scipy_zoom(*args, grid_mode, **kwargs):
    return _scipy_zoom(*args, **kwargs)


scipy_zoom = scipy_zoom if python_version()[:3] == '3.6' else _scipy_zoom


def _choose_cython_zoom(ndim: int, order: int, fast: bool) -> Callable:
    assert ndim <= 4, ndim
    assert order in (0, 1), order

    if ndim <= 3:
        if order == 0:
            return cython_fast_zoom3d_nearest if fast else cython_zoom3d_nearest

        return cython_fast_zoom3d_linear if fast else cython_zoom3d_linear

    if order == 0:
        return cython_fast_zoom4d_nearest if fast else cython_zoom4d_nearest

    return cython_fast_zoom4d_linear if fast else cython_zoom4d_linear


def _choose_numba_zoom(ndim: int, order: int) -> Callable:
    assert ndim <= 4, ndim
    assert order in (0, 1), order

    if ndim <= 3:
        if order == 0:
            from .src._numba_zoom import _zoom3d_nearest as numba_zoom
        else:
            from .src._numba_zoom import _zoom3d_linear as numba_zoom
    elif order == 0:
        from .src._numba_zoom import _zoom4d_nearest as numba_zoom
    else:
        from .src._numba_zoom import _zoom4d_linear as numba_zoom

    return numba_zoom


def zoom(
    x: np.ndarray,
    scale_factor: AxesParams,
    axis: AxesLike = None,
    order: int = 1,
    fill_value: Union[float, Callable] = 0,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Rescale `x` according to `scale_factor` along the `axis`.

    Uses a fast parallelizable implementation for fp32 / fp64 (and bool-int16-32-64 if order == 0) inputs,
    ndim <= 4 and order = 0 or 1.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array
    scale_factor: AxesParams
        float or sequence of floats describing how to scale along axes
    axis: AxesLike
        axis along which array will be scaled
    order: int
        order of interpolation
    fill_value: float | Callable
        value to fill past edges. If Callable (e.g. `numpy.min`) - `fill_value(x)` will be used
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `numba`, `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    zoomed: np.ndarray
        zoomed array

    Examples
    --------
    >>> zoomed = zoom(x, 2, axis=[0, 1])  # 3d array
    >>> zoomed = zoom(x, [1, 2, 3])  # different scales along each axes
    >>> zoomed = zoom(x.astype(int))  # will fall back to scipy's implementation because of int dtype
    """
    x = np.asarray(x)
    axis, scale_factor = broadcast_axis(axis, x.ndim, scale_factor)
    scale_factor = fill_by_indices(np.ones(x.ndim, 'float64'), scale_factor, axis)

    if callable(fill_value):
        fill_value = fill_value(x)

    # TODO: does `fill_value/cval` change anythng?
    return _zoom(x, scale_factor, order=order, cval=fill_value, num_threads=num_threads, backend=backend)


def zoom_to_shape(
    x: np.ndarray,
    shape: AxesLike,
    axis: AxesLike = None,
    order: int = 1,
    fill_value: Union[float, Callable] = 0,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Rescale `x` to match `shape` along the `axis`.

    Uses a fast parallelizable implementation for fp32 / fp64 (and bool-int16-32-64 if order == 0) inputs,
    ndim <= 4 and order = 0 or 1.

    Parameters
    ----------
    x: np.ndarray
        n-dimensional array
    shape: AxesLike
        float or sequence of floats describing desired lengths along axes
    axis: AxesLike
        axis along which array will be scaled
    order: int
        order of interpolation
    fill_value: float | Callable
        value to fill past edges. If Callable (e.g. `numpy.min`) - `fill_value(x)` will be used
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `numba`, `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    zoomed: np.ndarray
        zoomed array

    Examples
    --------
    >>> zoomed = zoom_to_shape(x, [3, 4, 5])  # 3d array
    >>> zoomed = zoom_to_shape(x, [6, 7], axis=[1, 2])  # zoom to shape along specified axes
    >>> zoomed = zoom_to_shape(x.astype(int))  # will fall back to scipy's implementation because of int dtype
    """
    x = np.asarray(x)
    axis, shape = broadcast_axis(axis, x.ndim, shape)
    old_shape = np.array(x.shape, 'float64')
    new_shape = np.array(fill_by_indices(x.shape, shape, axis), 'float64')

    return zoom(
        x,
        new_shape / old_shape,
        range(x.ndim),
        order=order,
        fill_value=fill_value,
        num_threads=num_threads,
        backend=backend,
    )


def _zoom(
    input: np.ndarray,
    zoom: Sequence[float],
    output: np.ndarray = None,
    order: int = 1,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
    *,
    grid_mode: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Faster parallelizable version of `scipy.ndimage.zoom` for fp32 / fp64 (and bool-int16-32-64 if order == 0) inputs.

    Works faster only for ndim <= 4. Shares interface with `scipy.ndimage.zoom`
    except for
    - `num_threads` argument defining how many threads to use (all available threads are used by default).
    - `backend` argument defining which backend to use. `numba`, `cython` and `scipy` are available,
        `cython` is used by default.

    See `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html`
    """
    backend = resolve_backend(backend, warn_stacklevel=4)
    if backend.name not in ('Scipy', 'Numba', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    ndim = input.ndim
    dtype = input.dtype
    cval = np.dtype(dtype).type(cval)
    zoom = fill_by_indices(np.ones(input.ndim, 'float64'), zoom, range(input.ndim))
    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=4)

    if backend.name == 'Scipy':
        return scipy_zoom(
            input, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter, grid_mode=grid_mode
        )

    if (
        (order not in (0, 1))
        or (
            dtype not in (np.float32, np.float64)
            if order == 1
            else dtype not in (bool, np.float32, np.float64, np.int16, np.int32, np.int64)
        )
        or ndim > 4
        or output is not None
        or mode != 'constant'
        or grid_mode
    ):
        warn(
            'Fast zoom is only supported for ndim<=4, dtype=fp32 or fp64 (and bool-int16-32-64 if order == 0), '
            "output=None, order=0 or 1, mode='constant', grid_mode=False. Falling back to scipy's implementation.",
        )
        return scipy_zoom(
            input, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter, grid_mode=grid_mode
        )

    if backend.name == 'Cython':
        src_zoom = _choose_cython_zoom(ndim, order, backend.fast)

    if backend.name == 'Numba':
        from numba import get_num_threads, njit, set_num_threads

        old_num_threads = get_num_threads()
        set_num_threads(num_threads)

        njit_kwargs = {kwarg: getattr(backend, kwarg) for kwarg in backend.__dataclass_fields__.keys()}
        src_zoom = njit(**njit_kwargs)(_choose_numba_zoom(ndim, order))

    n_dummy = 3 - ndim if ndim <= 3 else 0

    if n_dummy:
        input = input[(None,) * n_dummy]
        zoom = [*(1,) * n_dummy, *zoom]

    zoom = np.array(zoom, dtype=np.float64)
    is_contiguous = input.data.c_contiguous
    c_contiguous_permutaion = None
    args = () if backend.name in ('Numba',) else (num_threads,)

    if not is_contiguous:
        c_contiguous_permutaion = get_c_contiguous_permutaion(input)
        if c_contiguous_permutaion is not None:
            out = src_zoom(
                np.transpose(input, c_contiguous_permutaion),
                zoom[c_contiguous_permutaion],
                cval,
                *args,
            )
        else:
            warn("Input array can't be represented as C-contiguous, performance can drop a lot.")
            out = src_zoom(input, zoom, cval, *args)
    else:
        out = src_zoom(input, zoom, cval, *args)

    if c_contiguous_permutaion is not None:
        out = np.transpose(out, inverse_permutation(c_contiguous_permutaion))
    if n_dummy:
        out = out[(0,) * n_dummy]
    if backend.name == 'Numba':
        set_num_threads(old_num_threads)

    return out
