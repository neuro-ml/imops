from typing import Callable, Sequence, Tuple
from warnings import warn

import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from scipy.special import cosdg, sindg

from .backend import BackendLike, resolve_backend
from .src._fast_rotate import (
    _rotate3d_linear as cython_fast_rotate3d_linear,
    _rotate3d_nearest as cython_fast_rotate3d_nearest,
)
from .src._rotate import _rotate3d_linear as cython_rotate3d_linear, _rotate3d_nearest as cython_rotate3d_nearest
from .utils import normalize_num_threads


DTYPES = (np.float32, np.uint8, np.uint16, np.int16, np.int32)


def _rotation_plane(axes: Sequence[int], ndim: int) -> Tuple[int, int]:
    """The two distinct axes spanning the plane of rotation, normalized and sorted."""
    if len(axes) != 2:
        raise ValueError(f'`axes` must contain exactly two values, got {len(axes)}.')

    first, second = sorted(axis + ndim if axis < 0 else axis for axis in axes)
    if first < 0 or second >= ndim or first == second:
        raise ValueError(f'{tuple(axes)} is not a rotation plane of a {ndim}d array.')

    return first, second


def _plane_transform(in_plane_shape: Tuple[int, int], angle: float, reshape: bool):
    """Matrix, shift and output shape such that `source = matrix @ destination + shift`."""
    matrix = np.array([[cosdg(angle), sindg(angle)], [-sindg(angle), cosdg(angle)]])

    if reshape:
        rows, cols = in_plane_shape
        corners = matrix @ [[0, 0, rows, rows], [0, cols, 0, cols]]
        out_plane_shape = (np.ptp(corners, axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = np.array(in_plane_shape)

    shift = (np.array(in_plane_shape) - 1) / 2 - matrix @ ((out_plane_shape - 1) / 2)

    return matrix, shift, tuple(out_plane_shape)


def _fill_value(cval: float, dtype: np.dtype) -> np.generic:
    """`cval` cast to `dtype` the way scipy casts an interpolated value."""
    if dtype.kind == 'f':
        return dtype.type(cval)

    limits = np.iinfo(dtype)
    rounded = int(cval + 0.5 if cval > 0 else cval - 0.5)

    return dtype.type(min(max(rounded, limits.min), limits.max))


def _choose_cython_rotate(order: int, fast: bool) -> Callable:
    if order == 0:
        return cython_fast_rotate3d_nearest if fast else cython_rotate3d_nearest

    return cython_fast_rotate3d_linear if fast else cython_rotate3d_linear


def rotate(
    x: np.ndarray,
    angle: float,
    axes: Sequence[int] = (0, 1),
    reshape: bool = True,
    order: int = 1,
    cval: float = 0.0,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Rotate `x` by `angle` degrees in the plane spanned by `axes`.

    Faster parallelizable version of `scipy.ndimage.rotate` with `mode='constant'`, for order 0 or 1 and
    fp16-fp32-uint8-uint16-int16-int32 inputs. Anything else falls back to scipy.

    Parameters
    ----------
    x: np.ndarray
        array of at least 2 dimensions
    angle: float
        rotation angle in degrees
    axes: Sequence[int]
        the two axes that define the plane of rotation
    reshape: bool
        if True, the output grows to contain the whole rotated input
    order: int
        order of interpolation
    cval: float
        value to fill past the edges of the input
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    rotated: np.ndarray
        rotated array

    Examples
    --------
    ```python
    rotated = rotate(x, 30, axes=(1, 2))  # 3d array, every plane along axis 0 is rotated
    rotated = rotate(x, 30, axes=(1, 2), reshape=False)  # keeps the original shape
    rotated = rotate(mask, 30, axes=(1, 2), order=0)  # nearest neighbour, keeps the labels intact
    ```
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError(f'Input array must have at least 2 dimensions, got {x.ndim}.')

    axes = _rotation_plane(axes, x.ndim)
    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    if x.dtype == np.float16:
        as_float32 = rotate(x.astype(np.float32), angle, axes, reshape, order, cval, num_threads, backend)

        return as_float32.astype(np.float16)

    if backend.name == 'Scipy' or order not in (0, 1) or x.dtype not in DTYPES:
        if backend.name != 'Scipy':
            warn(
                'Fast rotate is only supported for order=0 or 1 and dtype=fp16-fp32-uint8-uint16-int16-int32. '
                "Falling back to scipy's implementation.",
                stacklevel=2,
            )

        return scipy_rotate(x, angle, axes=axes, reshape=reshape, order=order, cval=cval)

    matrix, shift, out_plane_shape = _plane_transform(tuple(x.shape[axis] for axis in axes), angle, reshape)
    planes = np.ascontiguousarray(np.moveaxis(x, axes, (-2, -1)))
    rotated = _choose_cython_rotate(order, backend.fast)(
        planes.reshape(-1, *planes.shape[-2:]),
        matrix,
        shift,
        *out_plane_shape,
        _fill_value(cval, x.dtype),
        num_threads,
    )

    return np.moveaxis(rotated.reshape(*planes.shape[:-2], *out_plane_shape), (-2, -1), axes)
