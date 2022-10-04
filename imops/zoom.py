from platform import python_version
from typing import Callable, Sequence, Union
from warnings import warn

import numpy as np
from scipy.ndimage import zoom as _scipy_zoom

from .backend import BackendLike, resolve_backend
from .src._fast_zoom import _zoom as cython_fast_zoom
from .src._zoom import _zoom as cython_zoom
from .utils import (
    FAST_MATH_WARNING,
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

    Uses a fast parallelizable implementation for fp32 / fp64 inputs, ndim <= 3 and order = 1.

    Parameters
    ----------
    x
    scale_factor
    axis
        axis along which the tensor will be scaled.
    order
        order of interpolation.
    fill_value
        value to fill past edges. If Callable (e.g. `numpy.min`) - `fill_value(x)` will be used.
    num_threads
        the number of threads to use for computation. Default = the cpu count.
    backend
        which backend to use. `numba`, `cython` and `scipy` are available, `cython` is used by default.
    """
    x = np.asarray(x)
    axis, scale_factor = broadcast_axis(axis, x.ndim, scale_factor)
    scale_factor = fill_by_indices(np.ones(x.ndim, 'float64'), scale_factor, axis)

    if callable(fill_value):
        fill_value = fill_value(x)

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

    Uses a fast parallelizable implementation for fp32 / fp64 inputs, ndim <= 3 and order = 1.

    Parameters
    ----------
    x
    shape
        final shape.
    axis
        axes along which the tensor will be scaled.
    order
        order of interpolation.
    fill_value
        value to fill past edges. If Callable (e.g. `numpy.min`) - `fill_value(x)` will be used.
    num_threads
        the number of threads to use for computation. Default = the cpu count.
    backend
        which backend to use. `numba`, `cython` and `scipy` are available, `cython` is used by default.
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
    Faster parallelizable version of `scipy.ndimage.zoom` for fp32 / fp64 inputs.

    Works faster only for ndim <= 3. Shares interface with `scipy.ndimage.zoom`
    except for
    - `num_threads` argument defining how many threads to use (all available threads are used by default).
    - `backend` argument defining which backend to use. `numba`, `cython` and `scipy` are available,
        `cython` is used by default.

    See `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html`
    """
    backend = resolve_backend(backend)
    if backend.name not in ('Scipy', 'Numba', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}"')

    ndim = input.ndim
    dtype = input.dtype
    zoom = fill_by_indices(np.ones(input.ndim, 'float64'), zoom, range(input.ndim))
    num_threads = normalize_num_threads(num_threads, backend)

    if backend.name == 'Scipy':
        return scipy_zoom(
            input, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter, grid_mode=grid_mode
        )

    if (
        dtype not in (np.float32, np.float64)
        or ndim > 3
        or output is not None
        or order != 1
        or mode != 'constant'
        or grid_mode
    ):
        warn(
            'Fast zoom is only supported for ndim<=3, dtype=float32 or float64, output=None, '
            "order=1, mode='constant', grid_mode=False. Falling back to scipy's implementation.",
        )

        return scipy_zoom(
            input, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter, grid_mode=grid_mode
        )

    if backend.name == 'Cython':
        if backend.fast:
            warn(FAST_MATH_WARNING)
            src_zoom = cython_fast_zoom
        else:
            src_zoom = cython_zoom

    if backend.name == 'Numba':
        from numba import get_num_threads, njit, set_num_threads

        from .src._numba_zoom import _zoom as numba_zoom

        old_num_threads = get_num_threads()
        set_num_threads(num_threads)

        njit_kwargs = {kwarg: getattr(backend, kwarg) for kwarg in backend.__dataclass_fields__.keys()}
        src_zoom = njit(**njit_kwargs)(numba_zoom)

    n_dummy = 3 - ndim

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
