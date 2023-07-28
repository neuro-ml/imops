from typing import Callable
from warnings import warn

import numpy as np
from scipy.ndimage import generate_binary_structure
from skimage.morphology import binary_dilation as scipy_binary_dilation, binary_erosion as scipy_binary_erosion

from .backend import BackendLike, Cython, Scipy, resolve_backend
from .src._fast_morphology import (
    _binary_dilation as cython_fast_binary_dilation,
    _binary_erosion as cython_fast_binary_erosion,
)
from .src._morphology import _binary_dilation as cython_binary_dilation, _binary_erosion as cython_binary_erosion
from .utils import composition_args, normalize_num_threads


def morphology_op_wrapper(
    op_name: str, backend2src_op: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
) -> Callable:
    def wrapped(
        image: np.ndarray, footprint: np.ndarray = None, num_threads: int = -1, backend: BackendLike = None
    ) -> np.ndarray:
        backend = resolve_backend(backend)
        if backend.name not in {x.name for x in backend2src_op.keys()}:
            raise ValueError(f'Unsupported backend "{backend.name}".')

        ndim = image.ndim
        num_threads = normalize_num_threads(num_threads, backend)

        if footprint is None:
            footprint = generate_binary_structure(ndim, 1)
        elif not footprint.size:
            raise RuntimeError('Footprint must not be empty.')

        if not image.any():
            warn(f'{op_name} is applied to the fully False mask (mask.any() == False).')
            return image

        if image.all():
            warn(f'{op_name} is applied to the fully True mask (mask.all() == True).')
            return image

        src_op = backend2src_op[backend]

        if backend.name == 'Scipy':
            return src_op(image, footprint)

        if ndim > 3:
            warn(
                f"Fast {' '.join(op_name.split('_'))} is only supported for ndim<=3. "
                "Falling back to scipy's implementation."
            )
            src_op(image, footprint)

        if footprint.ndim != ndim:
            raise ValueError('Input image and footprint number of dimensions must be the same.')

        n_dummy = 3 - ndim

        if n_dummy:
            image = image[(None,) * n_dummy]
            footprint = footprint[(None,) * n_dummy]

        out = src_op(image.astype(bool, copy=False), footprint.astype(bool, copy=False), num_threads)

        if n_dummy:
            out = out[(0,) * n_dummy]

        return out.astype(bool)

    return wrapped


def boxed_morphology(border_value: bool):
    def decorator(func):
        def wrapper(input, footprint, *args, **kwargs):
            if 'border_value' in kwargs and kwargs['border_value'] != border_value:
                return func(input, footprint, *args, **kwargs)
        
            if not footprint.any() or not input.any():
                return func(input, footprint, *args, **kwargs)
        
            box = mask_to_box(input)
            a_shape = np.asarray(input.shape)
            f_shape = np.asarray(footprint.shape)
        
            if border_value and (box[0] - f_shape < 0).any() or (box[1] + f_shape >= a_shape).any():
                return func(input, footprint, *args, **kwargs)

            kwargs['border_value'] = False
            
            orgn = f_shape // 2 - (1 - f_shape % 2)
            low = f_shape - orgn - 1
            high = orgn
            low = np.maximum(box[0] - low, 0)
            high = np.minimum(box[1] + high, a_shape)
            slices = tuple(map(slice, low, high))
            
            cropped = np.ascontiguousarray(input[slices])
            output_cropped = func(cropped, footprint, *args, **kwargs)
            output = np.zeros(input.shape, dtype=np.uint8)
            output[slices] = output_cropped
        
            return output

        return decorator
        
    return wrapper


_binary_dilation = morphology_op_wrapper(
    'binary_dilation',
    {
        Scipy(): scipy_binary_dilation,
        Cython(fast=False): cython_binary_dilation,
        Cython(fast=True): cython_fast_binary_dilation,
    },
)


def binary_dilation(
    image: np.ndarray,
    footprint: np.ndarray = None,
    num_threads: int = -1,
    border_value: bool = False
    backend: BackendLike = None
) -> np.ndarray:
    """
    Fast parallelizable binary morphological dilation of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    dilated: np.ndarray
        the result of morphological dilation

    Examples
    --------
    >>> dilated = binary_dilation(x)
    """

    return _binary_dilation(image, footprint, num_threads, backend)


_binary_erosion = morphology_op_wrapper(
    'binary_erosion',
    {
        Scipy(): scipy_binary_erosion,
        Cython(fast=False): cython_binary_erosion,
        Cython(fast=True): cython_fast_binary_erosion,
    },
)


boxed_binary_dilation = boxed_morphology(False)(binary_dilation)


def binary_erosion(
    image: np.ndarray,
    footprint: np.ndarray = None,
    num_threads: int = -1,
    border_value: bool = True,
    backend: BackendLike = None
) -> np.ndarray:
    """
    Fast parallelizable binary morphological erosion of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    eroded: np.ndarray
        the result of morphological erosion

    Examples
    --------
    >>> eroded = binary_erosion(x)
    """

    return _binary_erosion(image, footprint, num_threads, backend)


boxed_binary_erosion = boxed_morphology(True)(binary_erosion)


_binary_closing = morphology_op_wrapper(
    'binary_closing',
    {
        Scipy(): composition_args(scipy_binary_erosion, scipy_binary_dilation),
        Cython(fast=False): composition_args(cython_binary_erosion, cython_binary_dilation),
        Cython(fast=True): composition_args(cython_fast_binary_erosion, cython_fast_binary_dilation),
    },
)


def binary_closing(
    image: np.ndarray, footprint: np.ndarray = None, num_threads: int = -1, backend: BackendLike = None
) -> np.ndarray:
    """
    Fast parallelizable binary morphological closing of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    closing: np.ndarray
        the result of morphological closing

    Examples
    --------
    >>> closing = binary_closing(x)
    """

    return _binary_closing(image, footprint, num_threads, backend)


_binary_opening = morphology_op_wrapper(
    'binary_opening',
    {
        Scipy(): composition_args(scipy_binary_dilation, scipy_binary_erosion),
        Cython(fast=False): composition_args(cython_binary_dilation, cython_binary_erosion),
        Cython(fast=True): composition_args(cython_fast_binary_dilation, cython_fast_binary_erosion),
    },
)


def binary_opening(
    image: np.ndarray, footprint: np.ndarray = None, num_threads: int = -1, backend: BackendLike = None
) -> np.ndarray:
    """
    Fast parallelizable binary morphological opening of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    opening: np.ndarray
        the result of morphological opening

    Examples
    --------
    >>> opening = binary_opening(x)
    """

    return _binary_opening(image, footprint, num_threads, backend)
