from warnings import warn

import numpy as np

from .backend import BackendLike, resolve_backend
from .src._fast_numeric import (
    _parallel_pointwise_mul as cython_fast_parallel_pointwise_mul,
    _parallel_sum as cython_fast_parallel_sum,
)
from .src._numeric import _parallel_pointwise_mul as cython_parallel_pointwise_mul, _parallel_sum as cython_parallel_sum
from .utils import FAST_MATH_WARNING, normalize_num_threads


def parallel_sum(nums: np.ndarray, num_threads: int = -1, backend: BackendLike = None) -> float:
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
        which backend to use. Only `cython` is available

    Returns
    -------
    sum: float

    Examples
    --------
    >>> s = parallel_sum(x, num_threads=1)
    >>> s = parallel_sum(x, num_threads=8, backend=Cython(fast=True))  # ffast-math compiled version
    """
    backend = resolve_backend(backend)
    if backend.name not in ('Cython',):
        raise ValueError(f'Unsupported backend "{backend.name}"')

    num_threads = normalize_num_threads(num_threads, backend)

    if backend.name == 'Cython':
        if backend.fast:
            warn(FAST_MATH_WARNING)
            src_parallel_sum = cython_fast_parallel_sum
        else:
            src_parallel_sum = cython_parallel_sum

    return src_parallel_sum(nums, num_threads)


def parallel_pointwise_mul(
    nums1: np.ndarray, nums2: np.ndarray, num_threads: int = -1, backend: BackendLike = None
) -> np.ndarray:
    """
    Parallel pointwise multiplication of 2 flat numpy arrays

    Parameters
    ----------
    nums1: np.ndarray
        1-dimensional array
    nums2: np.ndarray
        1-dimensional array
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. Only `cython` is available

    Returns
    -------
    multiplied: np.ndarray
        result of nums1 * nums2

    Examples
    --------
    >>> mul = parallel_pointwise_mul(nums1, nums2, num_threads=8)
    >>> mul = parallel_pointwise_mul(nums1, nums2, backend=Cython(fast=True))  # ffast-math compiled version
    """
    if len(nums1) != len(nums2):
        raise ValueError('Both arrays must have the same length for pointswise multiplication.')

    backend = resolve_backend(backend)
    if backend.name not in ('Cython',):
        raise ValueError(f'Unsupported backend "{backend.name}"')

    num_threads = normalize_num_threads(num_threads, backend)

    if backend.name == 'Cython':
        if backend.fast:
            warn(FAST_MATH_WARNING)
            src_parallel_pointwise_mul = cython_fast_parallel_pointwise_mul
        else:
            src_parallel_pointwise_mul = cython_parallel_pointwise_mul

    return src_parallel_pointwise_mul(nums1, nums2, num_threads)
