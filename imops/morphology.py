from typing import Callable, Tuple, Union
from warnings import warn

import numpy as np
from edt import edt
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt, generate_binary_structure
from scipy.ndimage._nd_image import euclidean_feature_transform
from skimage.morphology import (
    binary_closing as scipy_binary_closing,
    binary_dilation as scipy_binary_dilation,
    binary_erosion as scipy_binary_erosion,
    binary_opening as scipy_binary_opening,
)

from .backend import BackendLike, Cython, Scipy, resolve_backend
from .box import add_margin, box_to_shape, mask_to_box, shape_to_box
from .compat import _ni_support
from .crop import crop_to_box
from .pad import restore_crop
from .src._fast_morphology import (
    _binary_dilation as cython_fast_binary_dilation,
    _binary_erosion as cython_fast_binary_erosion,
)
from .src._morphology import _binary_dilation as cython_binary_dilation, _binary_erosion as cython_binary_erosion
from .utils import morphology_composition_args, normalize_num_threads


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
        elif boxed:
            raise ValueError('`boxed==True` is incompatible with provided `output`')
        elif output.shape != image.shape:
            raise ValueError('Input image and output image shapes must be the same.')
        elif output.dtype != bool:
            raise ValueError(f'Output image must have `bool` dtype, got {output.dtype}.')
        elif not output.data.c_contiguous:
            # TODO: Implement morphology for `output` of arbitrary layout
            raise ValueError('`output` must be a C-contiguous array.')

        src_op = backend2src_op[backend]

        if backend.name == 'Scipy':
            if boxed:
                raise ValueError('`boxed==True` is incompatible with "Scipy" backend.')
            src_op(image, footprint, out=output)

            return output

        if ndim > 3:
            warn(
                f"Fast {' '.join(op_name.split('_'))} is only supported for ndim<=3. "
                "Falling back to scipy's implementation.",
                stacklevel=3,
            )
            backend2src_op[Scipy()](image, footprint, out=output)

            return output

        if footprint.ndim != image.ndim:
            raise ValueError('Input image and footprint number of dimensions must be the same.')

        if not image.any():
            warn(f'{op_name} is applied to the fully False mask (mask.any() == False).', stacklevel=3)  # noqa
            output.fill(False)

            return output

        if image.all():
            warn(f'{op_name} is applied to the fully True mask (mask.all() == True).', stacklevel=3)  # noqa
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

        return output

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
    ```python
    dilated = binary_dilation(x)
    ```
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
    ```python
    eroded = binary_erosion(x)
    ```
    """
    return _binary_erosion(image, footprint, output, boxed, num_threads, backend)


_binary_closing = morphology_op_wrapper(
    'binary_closing',
    {
        Scipy(): scipy_binary_closing,
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
    ```python
    closed = binary_closing(x)
    ```
    """

    return _binary_closing(image, footprint, output, boxed, num_threads, backend)


_binary_opening = morphology_op_wrapper(
    'binary_opening',
    {
        Scipy(): scipy_binary_opening,
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
    ```python
    opened = binary_opening(x)
    ```
    """

    return _binary_opening(image, footprint, output, boxed, num_threads, backend)


def distance_transform_edt(
    image: np.ndarray,
    sampling: Tuple[float] = None,
    return_distances: bool = True,
    return_indices: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Fast parallelizable Euclidean distance transform for <= 3D inputs

    This function calculates the distance transform of the `image`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    image : array_like
        input data to transform. Can be any type but will be converted
        into binary: 1 wherever input equates to True, 0 elsewhere
    sampling : tuple of `image.ndim` floats, optional
        spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied
    return_distances : bool, optional
        whether to calculate the distance transform.
        Default is True
    return_indices : bool, optional
        whether to calculate the feature transform.
        Default is False
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    distances : float32 ndarray, optional
        the calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied.
        It will have the same shape as the input array
    indices : int32 ndarray, optional
        the calculated feature transform. It has an input-shaped array for each
        dimension of the input. See example below.
        Returned only when `return_indices` is True and `indices` is not
        supplied

    Notes
    -----
    The Euclidean distance transform gives values of the Euclidean
    distance::

                    n
      y_i = sqrt(sum (x[i]-b[i])**2)
                    i

    where b[i] is the background point (value 0) with the smallest
    Euclidean distance to input points x[i], and n is the
    number of dimensions.

    Examples
    --------
    import numpy as np
    a = np.array(([0,1,1,1,1],
                  [0,0,1,1,1],
                  [0,1,1,1,1],
                  [0,1,1,1,0],
                  [0,1,1,0,0]))
    distance_transform_edt(a)
    array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
           [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
           [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    With a sampling of 2 units along x, 1 along y:

    distance_transform_edt(a, sampling=[2, 1])
    array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
           [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
           [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
           [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
           [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])

    Asking for indices as well:

    edt, inds = distance_transform_edt(a, return_indices=True)
    inds
    array([[[0, 0, 1, 1, 3],
            [1, 1, 1, 1, 3],
            [2, 2, 1, 3, 3],
            [3, 3, 4, 4, 3],
            [4, 4, 4, 4, 4]],
           [[0, 0, 1, 1, 4],
            [0, 1, 1, 1, 4],
            [0, 0, 1, 4, 4],
            [0, 0, 3, 3, 4],
            [0, 0, 3, 3, 4]]])
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    if backend.name == 'Scipy':
        return scipy_distance_transform_edt(image, sampling, return_distances, return_indices)

    if image.ndim > 3:
        warn("Fast Euclidean Distance Transform is only supported for ndim<=3. Falling back to scipy's implementation.")
        return scipy_distance_transform_edt(image, sampling, return_distances, return_indices)

    if (not return_distances) and (not return_indices):
        raise RuntimeError('At least one of `return_distances`/`return_indices` must be True')
    if image.dtype != bool:
        image = np.atleast_1d(np.where(image, 1, 0))
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, image.ndim)
        sampling = np.asarray(sampling, dtype=np.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    if return_indices:
        ft = np.zeros((image.ndim,) + image.shape, dtype=np.int32)
        euclidean_feature_transform(image, sampling, ft)

    if return_distances:
        if sampling is not None:
            dt = edt(image, anisotropy=sampling.astype(np.float32), parallel=num_threads)
        else:
            dt = edt(image, parallel=num_threads)

    result = []
    if return_distances:
        result.append(dt)
    if return_indices:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)

    if len(result) == 1:
        return result[0]

    return None
