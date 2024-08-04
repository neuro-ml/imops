import os
from contextlib import contextmanager
from itertools import permutations
from typing import Callable, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np

from .backend import BACKEND_NAME2ENV_NUM_THREADS_VAR_NAME, SINGLE_THREADED_BACKENDS, Backend, Cython
from .compat import normalize_axis_tuple
from .src._utils import _isin as cython_isin


AxesLike = Union[int, Sequence[int]]
AxesParams = Union[float, Sequence[float]]

ZOOM_SRC_DIM = 4
# TODO: define imops-specific environment variable like `OMP_NUM_THREADS`?
IMOPS_NUM_THREADS = None


def set_num_threads(num_threads: int) -> int:
    assert isinstance(num_threads, int) or num_threads is None, 'Number of threads must be int value or None.'
    global IMOPS_NUM_THREADS
    current = IMOPS_NUM_THREADS
    IMOPS_NUM_THREADS = num_threads
    return current


@contextmanager
def imops_num_threads(num_threads: int):
    previous = set_num_threads(num_threads)
    try:
        yield
    finally:
        set_num_threads(previous)


def normalize_num_threads(num_threads: int, backend: Backend, warn_stacklevel: int = 1) -> int:
    """Calculate the effective number of threads"""

    global IMOPS_NUM_THREADS
    if backend.name in SINGLE_THREADED_BACKENDS:
        if num_threads != -1:
            warn(
                f'"{backend.name}" backend is single-threaded. Setting `num_threads` has no effect.',
                stacklevel=warn_stacklevel,
            )
        return 1

    env_num_threads_var_name = BACKEND_NAME2ENV_NUM_THREADS_VAR_NAME[backend.name]
    # here we also handle the case `env_num_threads_var_name`=" " gracefully
    env_num_threads = os.environ.get(env_num_threads_var_name, '').strip()
    env_num_threads = int(env_num_threads) if env_num_threads else None
    # TODO: maybe let user set the absolute maximum number of threads?
    num_available_cpus = len(os.sched_getaffinity(0))

    max_num_threads = min(filter(bool, [IMOPS_NUM_THREADS, env_num_threads, num_available_cpus]))

    if num_threads >= 0:
        # FIXME
        if backend.name == 'Numba':
            warn(
                'Setting `num_threads` has no effect with "Numba" backend. '
                'Use `NUMBA_NUM_THREADS` environment variable.',
                stacklevel=warn_stacklevel,
            )
            return num_threads

        if num_threads > max_num_threads:
            if max_num_threads == IMOPS_NUM_THREADS:
                warn(
                    f'Required number of threads ({num_threads}) is greater than `IMOPS_NUM_THREADS` '
                    f'({IMOPS_NUM_THREADS}). Using {IMOPS_NUM_THREADS} threads.',
                    stacklevel=warn_stacklevel,
                )
            elif max_num_threads == env_num_threads:
                warn(
                    f'Required number of threads ({num_threads}) is greater than `{env_num_threads_var_name}` '
                    f'({env_num_threads}). Using {env_num_threads} threads.',
                    stacklevel=warn_stacklevel,
                )
            else:
                warn(
                    f'Required number of threads ({num_threads}) is greater than number of available CPU-s '
                    f'({num_available_cpus}). Using {num_available_cpus} threads.',
                    stacklevel=warn_stacklevel,
                )
        return min(num_threads, max_num_threads)

    return max_num_threads + num_threads + 1


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

    return normalize_axis_tuple(axis, dim, 'axis')


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


def morphology_composition_args(f, g) -> Callable:
    def wrapper(
        image: np.ndarray,
        footprint: np.ndarray,
        output: np.ndarray,
        num_threads: int,
    ):
        temp = np.empty_like(image, dtype=bool)
        temp = g(image, footprint, temp, num_threads)

        return f(temp, footprint, output, num_threads)

    return wrapper


def build_slices(start: Sequence[int], stop: Sequence[int] = None, step: Sequence[int] = None) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices built from `start` and `stop` with `step`.

    Examples
    --------
    ```python
    build_slices([1, 2, 3], [4, 5, 6])
    (slice(1, 4), slice(2, 5), slice(3, 6))
    build_slices([10, 11])
    (slice(10), slice(11))
    ```
    """

    check_len(*filter(lambda x: x is not None, [start, stop, step]))

    if stop is None and step is None:
        return tuple(map(slice, start))

    args = [
        start,
        stop if stop is not None else [None for _ in start],
        step if step is not None else [None for _ in start],
    ]

    return tuple(map(slice, *args))


def check_len(*args) -> None:
    lengths = list(map(len, args))
    if any(length != lengths[0] for length in lengths):
        raise ValueError(f'Arguments of equal length are required: {", ".join(map(str, lengths))}')


def assert_subdtype(dtype, ref_dtype, name):
    if not np.issubdtype(dtype, ref_dtype):
        raise ValueError(f'`{name}` must be of {ref_dtype.__name__} dtype, got {dtype}')


def isin(element: np.ndarray, test_elements: np.ndarray, num_threads: int = 1) -> np.ndarray:
    """
    Calculates `element in test_elements`, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True where
    an element of `element` is in `test_elements` and False otherwise.

    Parameters
    ----------
    element: np.ndarray
        n-dimensional array
    test_elements: np.ndarray
        1-d array of the values against which to test each value of element
    num_threads: int
        the number of threads to use for computation. Default = 1. If negative value passed
        cpu count + num_threads + 1 threads will be used

    Returns
    -------
    isin: np.ndarray, bool
        has the same shape as `element`. The values `element[isin]` are in `test_elements`

    Examples
    --------
    element = 2*np.arange(4).reshape((2, 2))
    test_elements = [1, 2, 4, 8]
    mask = isin(element, test_elements)
    """
    if element.dtype not in ('int16', 'int32', 'int64'):
        raise ValueError(f'Supported dtypes: int16, int32, int64, got {element.dtype}')

    num_threads = normalize_num_threads(num_threads, Cython(), warn_stacklevel=2)

    contiguos_element = np.ascontiguousarray(element)
    test_elements = np.asarray(test_elements, dtype=element.dtype)
    out = np.zeros_like(contiguos_element, dtype=bool)

    cython_isin(contiguos_element.ravel(), test_elements, out.ravel(), num_threads)

    return out


def make_immutable(array: np.ndarray) -> None:
    array.flags.writeable = False


def make_mutable(array: np.ndarray) -> None:
    array.flags.writeable = True
