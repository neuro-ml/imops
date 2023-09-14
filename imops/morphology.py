from typing import Callable
from warnings import warn

import numpy as np
from scipy.ndimage import generate_binary_structure
from skimage.morphology import binary_dilation as scipy_binary_dilation, binary_erosion as scipy_binary_erosion

from .backend import BackendLike, Cython, Scipy, resolve_backend
from .box import add_margin, box_to_shape, mask_to_box, shape_to_box
from .crop import crop_to_box
from .pad import restore_crop
from .src._fast_morphology import (
    _binary_dilation as cython_fast_binary_dilation,
    _binary_erosion as cython_fast_binary_erosion,
)
from .src._morphology import _binary_dilation as cython_binary_dilation, _binary_erosion as cython_binary_erosion
from .utils import composition_args, morphology_composition_args, normalize_num_threads


def morphology_op_wrapper(
    op_name: str, backend2src_op: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
) -> Callable:
    def wrapped(
        image: np.ndarray,
        footprint: np.ndarray = None,
        output: np.ndarray = None,
        boxed: bool = False,
        num_threads: int = -1,
        backend: BackendLike = None,
    ) -> np.ndarray:
        backend = resolve_backend(backend, warn_stacklevel=4)
        if backend.name not in {x.name for x in backend2src_op.keys()}:
            raise ValueError(f'Unsupported backend "{backend.name}".')

        ndim = image.ndim
        num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=4)

        if footprint is None:
            footprint = generate_binary_structure(ndim, 1)
        elif not footprint.size:
            raise RuntimeError('Footprint must not be empty.')

        if output is None:
            output = np.empty_like(image, dtype=bool)
        elif output.shape != image.shape:
            raise ValueError('Input image and output image shapes must be the same.')
        elif not output.data.c_contiguous:
            # TODO: Implement morphology for `output` of arbitrary layout
            raise ValueError('`output` must be a C-contiguous array.')

        src_op = backend2src_op[backend]

        if backend.name == 'Scipy':
            if boxed:
                raise ValueError('`boxed==True` is incompatible with "Scipy" backend.')
            output = src_op(image, footprint)

            return output

        if ndim > 3:
            warn(
                f"Fast {' '.join(op_name.split('_'))} is only supported for ndim<=3. "
                "Falling back to scipy's implementation.",
                stacklevel=3,
            )
            output = backend2src_op[Scipy()](image, footprint)

            return output

        if footprint.ndim != image.ndim:
            raise ValueError('Input image and footprint number of dimensions must be the same.')

        if not image.any():
            warn(f'{op_name} is applied to the fully False mask (mask.any() == False).', stacklevel=3)
            output.fill(False)

            return output

        if image.all():
            warn(f'{op_name} is applied to the fully True mask (mask.all() == True).', stacklevel=3)
            output.fill(True)

            return output

        n_dummy = 3 - ndim

        if n_dummy:
            image = image[(None,) * n_dummy]
            output = output[(None,) * n_dummy]
            footprint = footprint[(None,) * n_dummy]

        src_op_args = (image.astype(bool, copy=False), footprint.astype(bool, copy=False), output, num_threads)
        output = boxed_morphology(src_op, op_name)(*src_op_args) if boxed else src_op(*src_op_args)

        if n_dummy:
            output = output[(0,) * n_dummy]

        return output.astype(bool, copy=False)

    return wrapped


def boxed_morphology(func, op_name) -> Callable:
    # TODO: for consistency support exotic footprints which alter border pixels in Scikit-Image different from the
    # current implementation, e.g. footrint [[1, 1], [1, 0]] sets border pixel to 1 for `binary_erosion`
    def wrapped(
        image: np.ndarray,
        footprint: np.ndarray,
        output: np.ndarray,
        num_threads: int,
    ) -> np.ndarray:
        box_delta = np.asarray(footprint.shape) // 2

        image_box = shape_to_box(image.shape)
        tight_box = mask_to_box(image)
        supp_box = add_margin(tight_box, 2 * box_delta)

        # TODO: generalize to "anisotropic" images
        # TODO: make separate class for `Box` and implement comparison operators?
        if (supp_box[0] < image_box[0]).any() or (image_box[1] < supp_box[1]).any():
            return func(image, footprint, output, num_threads)

        final_crop_box = add_margin(tight_box, box_delta)

        supp_image = crop_to_box(image, supp_box)
        supp_output = np.empty_like(supp_image, dtype=bool)

        cropped = crop_to_box(
            func(supp_image, footprint, supp_output, num_threads),
            add_margin(shape_to_box(box_to_shape(supp_box)), -box_delta),  # crop border values of supp_box
        )

        output = restore_crop(cropped, final_crop_box, image.shape, False)

        return output

    return wrapped


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
    output: np.ndarray = None,
    boxed: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast parallelizable binary morphological dilation of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    output: np.ndarray
        array of the same shape as input, into which the output is placed (must be C-contiguous). By default, a new
        array is created
    boxed: bool
        if True, dilation is performed on cropped image which may speed up computation depedning on how localized True
        pixels are. This may induce differences with Scikit-Image implementation at border pixels if footprint is
        exotic (has even shape or center pixel is False)
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
    return _binary_dilation(image, footprint, output, boxed, num_threads, backend)


_binary_erosion = morphology_op_wrapper(
    'binary_erosion',
    {
        Scipy(): scipy_binary_erosion,
        Cython(fast=False): cython_binary_erosion,
        Cython(fast=True): cython_fast_binary_erosion,
    },
)


def binary_erosion(
    image: np.ndarray,
    footprint: np.ndarray = None,
    output: np.ndarray = None,
    boxed: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast parallelizable binary morphological erosion of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    output: np.ndarray
        array of the same shape as input, into which the output is placed (must be C-contiguous). By default, a new
        array is created
    boxed: bool
        if True, erosion is performed on cropped image which may speed up computation depedning on how localized True
        pixels are. This may induce differences with Scikit-Image implementation at border pixels if footprint is
        exotic (has even shape or center pixel is False)
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
    return _binary_erosion(image, footprint, output, boxed, num_threads, backend)


_binary_closing = morphology_op_wrapper(
    'binary_closing',
    {
        Scipy(): composition_args(scipy_binary_erosion, scipy_binary_dilation),
        Cython(fast=False): morphology_composition_args(cython_binary_erosion, cython_binary_dilation),
        Cython(fast=True): morphology_composition_args(cython_fast_binary_erosion, cython_fast_binary_dilation),
    },
)


def binary_closing(
    image: np.ndarray,
    footprint: np.ndarray = None,
    output: np.ndarray = None,
    boxed: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast parallelizable binary morphological closing of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    output: np.ndarray
        array of the same shape as input, into which the output is placed (must be C-contiguous). By default, a new
        array is created
    boxed: bool
        if True, closing is performed on cropped image which may speed up computation depedning on how localized True
        pixels are. This may induce differences with Scikit-Image implementation at border pixels if footprint is
        exotic (has even shape or center pixel is False)
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    closed: np.ndarray
        the result of morphological closing

    Examples
    --------
    >>> closed = binary_closing(x)
    """

    return _binary_closing(image, footprint, output, boxed, num_threads, backend)


_binary_opening = morphology_op_wrapper(
    'binary_opening',
    {
        Scipy(): composition_args(scipy_binary_dilation, scipy_binary_erosion),
        Cython(fast=False): morphology_composition_args(cython_binary_dilation, cython_binary_erosion),
        Cython(fast=True): morphology_composition_args(cython_fast_binary_dilation, cython_fast_binary_erosion),
    },
)


def binary_opening(
    image: np.ndarray,
    footprint: np.ndarray = None,
    output: np.ndarray = None,
    boxed: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast parallelizable binary morphological opening of an image

    Parameters
    ----------
    image: np.ndarray
        input image
    footprint: np.ndarray
        the neighborhood expressed as a n-D array of 1's and 0's. If None, use a cross-shaped footprint (connectivity=1)
    output: np.ndarray
        array of the same shape as input, into which the output is placed (must be C-contiguous). By default, a new
        array is created
    boxed: bool
        if True, opening is performed on cropped image which may speed up computation depedning on how localized True
        pixels are. This may induce differences with Scikit-Image implementation at border pixels if footprint is
        exotic (has even shape or center pixel is False)
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    opened: np.ndarray
        the result of morphological opening

    Examples
    --------
    >>> opened = binary_opening(x)
    """

    return _binary_opening(image, footprint, output, boxed, num_threads, backend)
