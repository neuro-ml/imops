import os
from typing import Callable, Sequence, Union
from warnings import warn

import numpy as np
from scipy.ndimage import zoom as scipy_zoom

from .src._fast_zoom import _zoom as src_zoom
from .utils import (
    AxesLike,
    AxesParams,
    broadcast_axis,
    fill_by_indices,
    get_c_contiguous_permutaion,
    inverse_permutation,
)


def zoom(
    x: np.ndarray,
    scale_factor: AxesParams,
    axis: AxesLike = None,
    order: int = 1,
    fill_value: Union[float, Callable] = 0,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Faster parallelizable version of `dpipe.im.shape_ops.zoom` for fp32 / fp64 inputs

    Works faster only for ndim <= 3. Shares interface with `dpipe.im.shape_ops.zoom`
    except for `num_threads` argument defining how many threads to use, all available threads are used by default.

    See `https://github.com/neuro-ml/deep_pipe/blob/master/dpipe/im/shape_ops.py#L19-L44`
    """
    x = np.asarray(x)
    axis, scale_factor = broadcast_axis(axis, x.ndim, scale_factor)
    scale_factor = fill_by_indices(np.ones(x.ndim, 'float64'), scale_factor, axis)

    if callable(fill_value):
        fill_value = fill_value(x)

    return _zoom(x, scale_factor, order=order, cval=fill_value, num_threads=num_threads)


def zoom_to_shape(
    x: np.ndarray,
    shape: AxesLike,
    axis: AxesLike = None,
    order: int = 1,
    fill_value: Union[float, Callable] = 0,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Faster parallelizable version of `dpipe.im.shape_ops.zoom_to_shape` for fp32 / fp64 inputs

    Works faster only for ndim <= 3. Shares interface with `dpipe.im.shape_ops.zoom_to_shape`
    except for `num_threads` argument defining how many threads to use, all available threads are used by default.

    See `https://github.com/neuro-ml/deep_pipe/blob/master/dpipe/im/shape_ops.py#L47-L68`
    """
    x = np.asarray(x)
    axis, shape = broadcast_axis(axis, x.ndim, shape)
    old_shape = np.array(x.shape, 'float64')
    new_shape = np.array(fill_by_indices(x.shape, shape, axis), 'float64')
    return zoom(x, new_shape / old_shape, range(x.ndim), order=order, fill_value=fill_value, num_threads=num_threads)


def _zoom(
    input: np.ndarray,
    zoom: Union[float, Sequence[float]],
    output: np.ndarray = None,
    order: int = 1,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
    *,
    grid_mode: bool = False,
    num_threads: int = -1,
) -> np.ndarray:
    """
    Faster parallelizable version of `scipy.ndimage.zoom` for fp32 / fp64 inputs

    Works faster only for ndim <= 3. Shares interface with `scipy.ndimage.zoom`
    except for `num_threads` argument defining how many threads to use, all available threads are used by default.

    See `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html`
    """
    if input.dtype not in (np.float32, np.float64):
        raise ValueError('Only fp32 and fp64 dtypes are allowed for zoom.')

    ndim = input.ndim

    if ndim > 3:
        return scipy_zoom(
            input, zoom, output=output, order=order, mode=mode, cval=cval, prefilter=prefilter, grid_mode=grid_mode
        )

    if output is not None:
        raise NotImplementedError('Only output=None is implemented for ndim <= 3.')
    if order != 1:
        raise NotImplementedError('Only 1-st order interpolation is implemented for ndim <= 3.')
    if mode != 'constant':
        raise NotImplementedError('Only constant mode is implemented for ndim <= 3.')
    if grid_mode is not False:
        raise NotImplementedError('Only grid_mode=False is implemented for ndim <= 3.')

    if isinstance(zoom, (Sequence, np.ndarray)):
        if len(zoom) != ndim:
            raise RuntimeError(f'Zoom argument {zoom} must be number or have length equal to the input rank {ndim}.')
    else:
        zoom = [zoom for _ in range(ndim)]

    num_threads = num_threads if num_threads != -1 else os.cpu_count()
    n_dummy = 3 - ndim

    if n_dummy:
        input = input[(None,) * n_dummy]
        zoom = [*(1,) * n_dummy, *zoom]

    is_contiguous = input.data.c_contiguous
    c_contiguous_permutaion = None

    if not is_contiguous:
        c_contiguous_permutaion = get_c_contiguous_permutaion(input)
        if c_contiguous_permutaion is not None:
            out = src_zoom(
                np.transpose(input, c_contiguous_permutaion),
                np.array(zoom, dtype=np.float64)[c_contiguous_permutaion],
                cval,
                num_threads,
            )
        else:
            warn("Input array can't be represented as C-contiguous, performance can drop a lot.")
            out = src_zoom(input, np.array(zoom, dtype=np.float64), cval, num_threads)
    else:
        out = src_zoom(input, np.array(zoom, dtype=np.float64), cval, num_threads)

    if c_contiguous_permutaion is not None:
        out = np.transpose(out, inverse_permutation(c_contiguous_permutaion))

    if n_dummy:
        out = out[(0,) * n_dummy]

    return out.astype(input.dtype, copy=False)
