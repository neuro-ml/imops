from typing import Callable, Sequence, Union

import numpy as np

from .backend import BackendLike, resolve_backend
from .src._fast_numeric import (
    _copy_3d as cython_fast_copy_3d,
    _copy_3d_fp16 as cython_fast_copy_3d_fp16,
    _copy_4d as cython_fast_copy_4d,
    _copy_4d_fp16 as cython_fast_copy_4d_fp16,
    _fill_3d as cython_fast_fill_3d,
    _fill_4d as cython_fast_fill_4d,
    _pointwise_add_array_3d as cython_fast_pointwise_add_array_3d,
    _pointwise_add_array_3d_fp16 as cython_fast_pointwise_add_array_3d_fp16,
    _pointwise_add_array_4d as cython_fast_pointwise_add_array_4d,
    _pointwise_add_array_4d_fp16 as cython_fast_pointwise_add_array_4d_fp16,
    _pointwise_add_value_3d as cython_fast_pointwise_add_value_3d,
    _pointwise_add_value_3d_fp16 as cython_fast_pointwise_add_value_3d_fp16,
    _pointwise_add_value_4d as cython_fast_pointwise_add_value_4d,
    _pointwise_add_value_4d_fp16 as cython_fast_pointwise_add_value_4d_fp16,
)
from .src._numeric import (
    _copy_3d as cython_copy_3d,
    _copy_3d_fp16 as cython_copy_3d_fp16,
    _copy_4d as cython_copy_4d,
    _copy_4d_fp16 as cython_copy_4d_fp16,
    _fill_3d as cython_fill_3d,
    _fill_4d as cython_fill_4d,
    _pointwise_add_array_3d as cython_pointwise_add_array_3d,
    _pointwise_add_array_3d_fp16 as cython_pointwise_add_array_3d_fp16,
    _pointwise_add_array_4d as cython_pointwise_add_array_4d,
    _pointwise_add_array_4d_fp16 as cython_pointwise_add_array_4d_fp16,
    _pointwise_add_value_3d as cython_pointwise_add_value_3d,
    _pointwise_add_value_3d_fp16 as cython_pointwise_add_value_3d_fp16,
    _pointwise_add_value_4d as cython_pointwise_add_value_4d,
    _pointwise_add_value_4d_fp16 as cython_pointwise_add_value_4d_fp16,
)
from .utils import normalize_num_threads


_TYPES = (np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
_STR_TYPES = ('int16', 'int32', 'int64', 'float16', 'float32', 'float64')
# TODO: Decide which value to use. Functions below are quite fast and simple, so parallelization overhead is noticeable.
_NUMERIC_DEFAULT_NUM_THREADS = 4


# TODO: Maybe dict is better?
def _choose_cython_pointwise_add(ndim: int, summand_is_array: bool, is_fp16: bool, fast: bool) -> Callable:
    assert ndim <= 4, ndim

    if ndim <= 3:
        if summand_is_array:
            if is_fp16:
                return cython_fast_pointwise_add_array_3d_fp16 if fast else cython_pointwise_add_array_3d_fp16

            return cython_fast_pointwise_add_array_3d if fast else cython_pointwise_add_array_3d

        if is_fp16:
            return cython_fast_pointwise_add_value_3d_fp16 if fast else cython_pointwise_add_value_3d_fp16

        return cython_fast_pointwise_add_value_3d if fast else cython_pointwise_add_value_3d

    if summand_is_array:
        if is_fp16:
            return cython_fast_pointwise_add_array_4d_fp16 if fast else cython_pointwise_add_array_4d_fp16

        return cython_fast_pointwise_add_array_4d if fast else cython_pointwise_add_array_4d

    if is_fp16:
        return cython_fast_pointwise_add_value_4d_fp16 if fast else cython_pointwise_add_value_4d_fp16

    return cython_fast_pointwise_add_value_4d if fast else cython_pointwise_add_value_4d


def _choose_cython_fill_(ndim: int, fast: bool) -> Callable:
    assert ndim <= 4, ndim

    if ndim <= 3:
        return cython_fast_fill_3d if fast else cython_fill_3d

    return cython_fast_fill_4d if fast else cython_fill_4d


def _choose_cython_copy(ndim: int, is_fp16: bool, fast: bool) -> Callable:
    assert ndim <= 4, ndim

    if ndim <= 3:
        if is_fp16:
            return cython_fast_copy_3d_fp16 if fast else cython_copy_3d_fp16

        return cython_fast_copy_3d if fast else cython_copy_3d

    if is_fp16:
        return cython_fast_copy_4d_fp16 if fast else cython_copy_4d_fp16

    return cython_fast_copy_4d if fast else cython_copy_4d


def pointwise_add(
    nums: np.ndarray,
    summand: Union[np.array, int, float],
    output: np.ndarray = None,
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Perform pointwise addition between array and array or scalar.

    Uses a fast parallelizable implementation for fp16-32-64 and int16-32-64 inputs and ndim <= 4.

    Parameters
    ----------
    nums: np.ndarray
        n-dimensional array
    summand: np.ndarray | int | float
        array of the same shape or scalar
    output: np.ndarray
        array of the same shape as input, into which the output is placed. By default, a new
        array is created
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    sum: np.ndarray
        result of summation

    Examples
    --------
    >>> sum = pointwise_add(x, 1, x)  # inplace addition
    >>> sum = pointwise_add(x, 1, backend='Scipy')  # just `np.add`
    >>> sum = pointwise_add(x.astype('float32'), x.astype('float16'))  # will fail because of different dtypes
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    dtype = nums.dtype

    if dtype not in _TYPES:
        raise ValueError(f'Input array dtype must be one of {", ".join(_STR_TYPES)}, got {dtype}.')

    if output is None:
        output = np.empty_like(nums, dtype=dtype)
    elif output.shape != nums.shape:
        raise ValueError(f'Input array and output array shapes must be the same, got {nums.shape} vs {output.shape}.')
    elif dtype != output.dtype:
        raise ValueError(f'Input array and output array dtypes must be the same, got {dtype} vs {output.dtype}.')

    summand_is_array = isinstance(summand, np.ndarray)
    if summand_is_array:
        if dtype != summand.dtype:
            raise ValueError(f'Input and summand arrays must have same dtypes, got {dtype} vs {summand.dtype}.')
    elif not isinstance(summand, (*_TYPES, *(int, float))):
        raise ValueError(f'Summand dtype must be one of {", ".join(_STR_TYPES)}, got {type(summand)}.')
    else:
        summand = dtype.type(summand)

    ndim = nums.ndim
    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    if backend.name == 'Scipy' or ndim > 4:
        np.add(nums, summand, out=output)
        return output

    is_fp16 = dtype == np.float16
    src_pointwise_add = _choose_cython_pointwise_add(ndim, summand_is_array, is_fp16, backend.fast)

    n_dummy = 3 - ndim if ndim <= 3 else 0

    if n_dummy:
        nums = nums[(None,) * n_dummy]
        output = output[(None,) * n_dummy]
        if summand_is_array:
            summand = summand[(None,) * n_dummy]

    if is_fp16:
        output = src_pointwise_add(
            nums.view(np.uint16), summand.view(np.uint16), output.view(np.uint16), num_threads
        ).view(np.float16)
    else:
        output = src_pointwise_add(nums, summand, output, num_threads)

    if n_dummy:
        output = output[(0,) * n_dummy]

    return output


def fill_(
    nums: np.ndarray,
    value: Union[np.number, int, float],
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> None:
    """
    Fill the array with a scalar value.

    Uses a fast parallelizable implementation for fp16-32-64 and int16-32-64 inputs and ndim <= 4.

    Parameters
    ----------
    nums: np.ndarray
        n-dimensional array
    value: np.number | int | float
        scalar
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Examples
    --------
    >>> fill_(x, 1)
    >>> fill_(np.empty((2, 3, 4)), 42)
    >>> fill_(x.astype('uint16'), 3)  # will fail because of unsupported uint16 dtype
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    ndim = nums.ndim
    dtype = nums.dtype

    if dtype not in _TYPES or backend.name == 'Scipy' or ndim > 4:
        nums.fill(value)
        return

    is_fp16 = dtype == np.float16
    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)
    src_fill_ = _choose_cython_fill_(ndim, backend.fast)
    value = dtype.type(value)

    n_dummy = 3 - ndim if ndim <= 3 else 0

    if n_dummy:
        nums = nums[(None,) * n_dummy]

    if is_fp16:
        src_fill_(nums.view(np.uint16), value.view(np.uint16), num_threads)
    else:
        src_fill_(nums, value, num_threads)

    if n_dummy:
        nums = nums[(0,) * n_dummy]


def full(
    shape: Union[int, Sequence[int]],
    fill_value: Union[np.number, int, float],
    dtype: Union[type, str] = None,
    order: str = 'C',
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Uses a fast parallelizable implementation for fp16-32-64 and int16-32-64 inputs and ndim <= 4.

    Parameters
    ----------
    shape: int | Sequence[int]
        desired shape
    fill_value: np.number | int | float
        scalar to fill array with
    dtype: type | str
        desired dtype to which `fill_value` will be casted. If not specified, `np.array(fill_value).dtype` will be used
    order: str
        whether to store multidimensional data in C or F contiguous order in memory
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Examples
    --------
    >>> x = full((2, 3, 4), 1.0)  # same as `np.ones((2, 3, 4))`
    >>> x = full((2, 3, 4), 1.5, dtype=int)  # same as np.ones((2, 3, 4), dtype=int)
    >>> x = full((2, 3, 4), 1, dtype='uint16')  # will fail because of unsupported uint16 dtype
    """
    nums = np.empty(shape, dtype=dtype, order=order)

    if dtype is not None:
        fill_value = nums.dtype.type(fill_value)

    fill_(nums, fill_value, num_threads, backend)

    return nums


def copy(
    nums: np.ndarray,
    output: np.ndarray = None,
    order: str = 'K',
    num_threads: int = _NUMERIC_DEFAULT_NUM_THREADS,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Return copy of the given array.

    Uses a fast parallelizable implementation for fp16-32-64 and int16-32-64 inputs and ndim <= 4.

    Parameters
    ----------
    nums: np.ndarray
        n-dimensional array
    output: np.ndarray
        array of the same shape and dtype as input, into which the copy is placed. By default, a new
        array is created
    order: str
        controls the memory layout of the copy. `C` means C-order, `F` means F-order, `A` means `F` if a is Fortran
        contiguous, `C` otherwise. `K` means match the layout of a as closely as possible
    num_threads: int
        the number of threads to use for computation. Default = 4. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    copy: np.ndarray
        copy of array

    Examples
    --------
    >>> copied = copy(x)
    >>> copied = copy(x, backend='Scipy')  # same as `np.copy`
    >>> copy(x, output=y)  # copied into `y`
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    ndim = nums.ndim
    dtype = nums.dtype
    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    if output is None:
        output = np.empty_like(nums, dtype=dtype, order=order)
    elif output.shape != nums.shape:
        raise ValueError(f'Input array and output array shapes must be the same, got {nums.shape} vs {output.shape}.')
    elif dtype != output.dtype:
        raise ValueError(f'Input array and output array dtypes must be the same, got {dtype} vs {output.dtype}.')

    if dtype not in _TYPES or backend.name == 'Scipy' or ndim > 4:
        output = np.copy(nums, order=order)
        return output

    is_fp16 = dtype == np.float16
    src_copy = _choose_cython_copy(ndim, is_fp16, backend.fast)

    n_dummy = 3 - ndim if ndim <= 3 else 0

    if n_dummy:
        nums = nums[(None,) * n_dummy]
        output = output[(None,) * n_dummy]

    if is_fp16:
        src_copy(nums.view(np.uint16), output.view(np.uint16), num_threads)
    else:
        src_copy(nums, output, num_threads)

    if n_dummy:
        nums = nums[(0,) * n_dummy]
        output = output[(0,) * n_dummy]

    return output


# TODO: add parallel astype?
