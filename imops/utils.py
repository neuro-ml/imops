import os
from itertools import permutations
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np

from .backend import BACKEND2NUM_THREADS_VAR_NAME, SINGLE_THREADED_BACKENDS, Backend


AxesLike = Union[int, Sequence[int]]
AxesParams = Union[float, Sequence[float]]

FAST_MATH_WARNING = (
    'Be careful, `fast=True` is an experimental feature. It enables some dangerous optimizations which can lead to '
    'unexpected results, use at your own risk! Visit https://simonbyrne.github.io/notes/fastmath/ for more information.'
)


def normalize_num_threads(num_threads: int, backend: Backend):
    if backend.name in SINGLE_THREADED_BACKENDS:
        if num_threads != -1:
            warn(f'"{backend.name}" backend is single-threaded. Setting `num_threads` has no effect.')
        return 1
    if num_threads >= 0:
        # FIXME
        if backend.name == 'Numba':
            warn(
                'Setting `num_threads` has no effect with "Numba" backend. '
                'Use `NUMBA_NUM_THREADS` environment variable.'
            )
        return num_threads

    num_threads_var_name = BACKEND2NUM_THREADS_VAR_NAME[backend.name]
    # here we also handle the case `num_threads_var`=" " gracefully
    env_num_threads = os.environ.get(num_threads_var_name, '').strip()
    max_threads = int(env_num_threads) if env_num_threads else len(os.sched_getaffinity(0))

    return max_threads + num_threads + 1


def normalize_axes(x: np.ndarray, axes) -> np.ndarray:
    assert x.ndim in [2, 3]
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
        if axes is None:
            axes = [1, 2]
        else:
            axes = np.array(np.core.numeric.normalize_axis_tuple(axes, 2)) + 1

    if axes is None:
        raise ValueError('For 3D inputs must pass the `axes` argument.')

    return np.moveaxis(x, axes, [1, 2]), axes, squeeze


def restore_axes(x: np.ndarray, axes, squeeze: bool) -> np.ndarray:
    x = np.moveaxis(x, [1, 2], axes)
    if squeeze:
        (x,) = x

    return x


def get_c_contiguous_permutaion(array: np.ndarray) -> Optional[np.ndarray]:
    for permutation in permutations(range(array.ndim)):
        if np.transpose(array, permutation).data.c_contiguous:
            return np.array(permutation)

    return None


def inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    inverse_permutation = np.arange(permutation.shape[0])
    inverse_permutation[permutation] = inverse_permutation.copy()

    return inverse_permutation


def axis_from_dim(axis: Union[AxesLike, None], dim: int) -> tuple:
    if axis is None:
        return tuple(range(dim))

    return np.core.numeric.normalize_axis_tuple(axis, dim, 'axis')


def broadcast_axis(axis: Union[AxesLike, None], dim: int, *values: Union[AxesLike, AxesParams]):
    axis = axis_from_dim(axis, dim)
    values = [to_axis(axis, x) for x in values]
    sizes = set(map(len, values))
    if not sizes <= {len(axis)}:
        raise ValueError(f"Params sizes don't match with the axes: {axis} vs {sizes}.")

    return (axis, *values)


def to_axis(axis, value):
    value = np.atleast_1d(value)
    if len(value) == 1:
        value = np.repeat(value, len(axis), 0)

    return value


def fill_by_indices(target, values, indices):
    target = np.array(target)
    target[list(indices)] = values

    return tuple(target)


def broadcast_to_axis(axis: AxesLike, *arrays: AxesParams):
    if not arrays:
        raise ValueError('No arrays provided.')

    arrays = list(map(np.atleast_1d, arrays))
    lengths = list(map(len, arrays))
    if axis is None:
        raise ValueError('`axis` cannot be None.')

    if not all(len(axis) == x or x == 1 for x in lengths):
        raise ValueError(f'Axes and arrays are not broadcastable: {len(axis)} vs {", ".join(map(str, lengths))}.')

    return tuple(np.repeat(x, len(axis) // len(x), 0) for x in arrays)
