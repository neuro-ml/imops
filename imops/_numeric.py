from typing import Callable, Union
from warnings import warn

import numpy as np

from .backend import BackendLike, resolve_backend
from .src._fast_numeric import (
    _pointwise_add_array_3d as cython_fast_pointwise_add_array_3d,
    _pointwise_add_array_4d as cython_fast_pointwise_add_array_4d,
    _pointwise_add_value_3d as cython_fast_pointwise_add_value_3d,
    _pointwise_add_value_4d as cython_fast_pointwise_add_value_4d,
    _pointwise_mul_3d as cython_fast_pointwise_mul_3d,
    _sum_1d as cython_fast_sum_1d,
)
from .src._numeric import (
    _pointwise_add_array_3d as cython_pointwise_add_array_3d,
    _pointwise_add_array_4d as cython_pointwise_add_array_4d,
    _pointwise_add_value_3d as cython_pointwise_add_value_3d,
    _pointwise_add_value_4d as cython_pointwise_add_value_4d,
    _pointwise_mul_3d as cython_pointwise_mul_3d,
    _sum_1d as cython_sum_1d,
)
from .utils import normalize_num_threads


TYPES = (np.int16, np.int32, np.int64, np.float32, np.float64)


def _sum(nums: np.ndarray, num_threads: int = -1, backend: BackendLike = None) -> float:
    """
    Parallel sum of flat numpy array

    Parameters
    ----------
    nums: np.ndarray
        1-dimensional array
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` (primarly for benchmarking) are available,
        `cython` is used by default

    Returns
    -------
    sum: float

    Examples
    --------
    >>> s = _sum(x, num_threads=1)
    >>> s = _sum(x, num_threads=8, backend=Cython(fast=True))  # ffast-math compiled version
    """
    ndim = nums.ndim

    if ndim != 1:
        raise ValueError(f'Input must be 1-dimensional instead of {ndim}-dimensional.')

    backend = resolve_backend(backend)
    if backend.name not in ('Cython', 'Scipy'):
        raise ValueError(f'Unsupported backend "{backend.name}"')

    num_threads = normalize_num_threads(num_threads, backend)

    if backend.name == 'Scipy':
        return nums.sum()

    if backend.name == 'Cython':
        src_sum_1d = cython_fast_sum_1d if backend.fast else cython_sum_1d

    return src_sum_1d(nums, num_threads)


def _mul(nums1: np.ndarray, nums2: np.ndarray, num_threads: int = -1, backend: BackendLike = None) -> np.ndarray:
    """
    Parallel pointwise multiplication of 2 numpy arrays (aka x * y). Works faster only for ndim <= 3.

    Parameters
    ----------
    nums1: np.ndarray
    nums2: np.ndarray
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` (primarly for benchmarking) are available,
        `cython` is used by default

    Returns
    -------
    multiplied: np.ndarray
        result of nums1 * nums2

    Examples
    --------
    >>> mul = _mul(nums1, nums2, num_threads=8)
    >>> mul = _mul(np.ones((2, 3)), np.ones((1, 3)))  # broadcasting, mul.shape == (2, 3)
    >>> mul = _mul(nums1, nums2, backend=Cython(fast=True))  # ffast-math compiled version
    """
    if not nums1.size and not nums2.size:
        return np.array([], dtype=nums1.dtype)
    if bool(nums1.size) ^ bool(nums2.size):
        raise ValueError('One of the arrays is empty, hence pointwise multiplication cannot be performed.')
    if len(nums1.shape) != len(nums2.shape):
        raise ValueError('Both arrays must have the same number of dimensions for pointwise multiplication.')
    for dim1, dim2 in zip(nums1.shape, nums2.shape):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise ValueError(f'Arrays of shapes {nums1.shape} and {nums2.shape} are not broadcastable.')

    if nums1.ndim > 3:
        warn('Parallel pointwise multiplication is only supported for ndim<=3. Falling back to naive x * y.')

        return nums1 * nums2

    backend = resolve_backend(backend)
    if backend.name not in ('Cython', 'Scipy'):
        raise ValueError(f'Unsupported backend "{backend.name}"')

    num_threads = normalize_num_threads(num_threads, backend)

    if backend.name == 'Scipy':
        return nums1 * nums2

    if backend.name == 'Cython':
        src_pointwise_mul_3d = cython_fast_pointwise_mul_3d if backend.fast else cython_pointwise_mul_3d

    n_dummy = 3 - nums1.ndim

    if n_dummy:
        nums1 = nums1[(None,) * n_dummy]
        nums2 = nums2[(None,) * n_dummy]

    out = src_pointwise_mul_3d(nums1, nums2, np.maximum(nums1.shape, nums2.shape), num_threads)

    if n_dummy:
        out = out[(0,) * n_dummy]

    return out


def _choose_cython_pointwise_add(ndim: int, summand_is_array: bool, fast: bool) -> Callable:
    assert ndim <= 4, ndim

    if ndim <= 3:
        if summand_is_array:
            return cython_fast_pointwise_add_array_3d if fast else cython_pointwise_add_array_3d
        return cython_fast_pointwise_add_value_3d if fast else cython_pointwise_add_value_3d

    if summand_is_array:
        return cython_fast_pointwise_add_array_4d if fast else cython_pointwise_add_array_4d

    return cython_fast_pointwise_add_value_4d if fast else cython_pointwise_add_value_4d


def pointwise_add(
    nums: np.ndarray,
    summand: Union[np.array, float],
    output: np.ndarray = None,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    backend = resolve_backend(backend)
    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    ndim = nums.ndim
    dtype = nums.dtype

    if dtype not in TYPES:
        raise ValueError(f'Input array dtype must be one of {", ".join(TYPES)}, got {dtype}.')

    if output is None:
        output = np.empty_like(nums, dtype=dtype)
    elif output.shape != nums.shape:
        raise ValueError('Input array and output array shapes must be the same.')
    elif dtype != output.dtype:
        raise ValueError('Input array and output array dtypes must be the same.')

    summand_is_array = isinstance(summand, np.ndarray)
    if summand_is_array:
        if dtype != summand.dtype:
            raise ValueError(f'Input and summand arrays must have same dtypes, got {dtype} vs {summand.dtype}.')
    elif not isinstance(summand, (*TYPES, *(int, float))):
        raise ValueError(f'Summand dtype must be one of {", ".join(TYPES)}, got {type(summand)}.')
    else:
        summand = nums.dtype.type(summand)

    if backend.name == 'Scipy' or ndim > 4:
        np.add(nums, summand, out=output)
        return output

    num_threads = normalize_num_threads(num_threads, backend)
    src_pointwise_add = _choose_cython_pointwise_add(ndim, summand_is_array, backend.fast)

    n_dummy = 3 - ndim if ndim <= 3 else 0

    if n_dummy:
        nums = nums[(None,) * n_dummy]
        output = output[(None,) * n_dummy]
        if summand_is_array:
            summand = summand[(None,) * n_dummy]

    output = src_pointwise_add(nums, summand, output, num_threads)

    if n_dummy:
        output = output[(0,) * n_dummy]

    return output
