from typing import Callable, Sequence, Union
from warnings import warn

import numpy as np
from scipy.ndimage import zoom as scipy_zoom

from .src._fast_zoom import _zoom as fast_src_zoom
from .src._zoom import _zoom as src_zoom
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


def zoom(
    x: np.ndarray,
    scale_factor: AxesParams,
    axis: AxesLike = None,
    order: int = 1,
    fill_value: Union[float, Callable] = 0,
    num_threads: int = -1,
    fast: bool = False,
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
    fast
        whether to use `-ffast-math` compiled version (almost no effect for zoom).
    """
    x = np.asarray(x)
    axis, scale_factor = broadcast_axis(axis, x.ndim, scale_factor)
    scale_factor = fill_by_indices(np.ones(x.ndim, 'float64'), scale_factor, axis)

    if callable(fill_value):
        fill_value = fill_value(x)

    return _zoom(x, scale_factor, order=order, cval=fill_value, num_threads=num_threads, fast=fast)


def zoom_to_shape(
    x: np.ndarray,
    shape: AxesLike,
    axis: AxesLike = None,
    order: int = 1,
    fill_value: Union[float, Callable] = 0,
    num_threads: int = -1,
    fast: bool = False,
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
    fast
        whether to use `-ffast-math` compiled version (almost no effect for zoom).
    """
    x = np.asarray(x)
    axis, shape = broadcast_axis(axis, x.ndim, shape)
    old_shape = np.array(x.shape, 'float64')
    new_shape = np.array(fill_by_indices(x.shape, shape, axis), 'float64')

    return zoom(
        x, new_shape / old_shape, range(x.ndim), order=order, fill_value=fill_value, num_threads=num_threads, fast=fast
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
    fast: bool = False,
) -> np.ndarray:
    """
    Faster parallelizable version of `scipy.ndimage.zoom` for fp32 / fp64 inputs

    Works faster only for ndim <= 3. Shares interface with `scipy.ndimage.zoom`
    except for `num_threads` argument defining how many threads to use (all available threads are used by default)
    and `fast` argument defining whether to use `-ffast-math` compiled version or not (almost no effect for zoom).

    See `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html`
    """
    ndim = input.ndim
    zoom = fill_by_indices(np.ones(input.ndim, 'float64'), zoom, range(input.ndim))

    if (
        input.dtype not in (np.float32, np.float64)
        or ndim > 3
        or output is not None
        or order != 1
        or mode != 'constant'
        or grid_mode
    ):
        warn(
            'Fast zoom is only supported for ndim<=3, dtype=float32 or float64, output=None, '
            "order=1, mode='constant', grid_mode=False. Falling back to scipy's implementation",
            UserWarning,
        )

        return scipy_zoom(
            input, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter, grid_mode=grid_mode
        )

    if fast:
        warn(FAST_MATH_WARNING, UserWarning)
        src_zoom_ = fast_src_zoom
    else:
        src_zoom_ = src_zoom

    num_threads = normalize_num_threads(num_threads)

    n_dummy = 3 - ndim

    if n_dummy:
        input = input[(None,) * n_dummy]
        zoom = [*(1,) * n_dummy, *zoom]

    is_contiguous = input.data.c_contiguous
    c_contiguous_permutaion = None

    if not is_contiguous:
        c_contiguous_permutaion = get_c_contiguous_permutaion(input)
        if c_contiguous_permutaion is not None:
            out = src_zoom_(
                np.transpose(input, c_contiguous_permutaion),
                np.array(zoom, dtype=np.float64)[c_contiguous_permutaion],
                cval,
                num_threads,
            )
        else:
            warn("Input array can't be represented as C-contiguous, performance can drop a lot.")
            out = src_zoom_(input, np.array(zoom, dtype=np.float64), cval, num_threads)
    else:
        out = src_zoom_(input, np.array(zoom, dtype=np.float64), cval, num_threads)

    if c_contiguous_permutaion is not None:
        out = np.transpose(out, inverse_permutation(c_contiguous_permutaion))

    if n_dummy:
        out = out[(0,) * n_dummy]

    return out.astype(input.dtype, copy=False)
