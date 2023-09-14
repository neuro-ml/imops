from collections import namedtuple
from platform import python_version
from typing import List, NamedTuple, Sequence, Tuple, Union
from warnings import warn

import numpy as np
from cc3d import connected_components
from fastremap import remap, unique
from scipy.ndimage import center_of_mass as scipy_center_of_mass
from skimage.measure import label as skimage_label

from .backend import BackendLike, resolve_backend
from .src._fast_measure import (
    _center_of_mass as _fast_center_of_mass,
    _labeled_center_of_mass as _fast_labeled_center_of_mass,
)
from .src._measure import _center_of_mass, _labeled_center_of_mass
from .utils import normalize_num_threads


# (ndim, skimage_connectivity) -> cc3d_connectivity
_SKIMAGE2CC3D = {
    (1, 1): 4,
    (2, 1): 4,
    (2, 2): 8,
    (3, 1): 6,
    (3, 2): 18,
    (3, 3): 26,
}


def label(
    label_image: np.ndarray,
    background: int = None,
    connectivity: int = None,
    return_num: bool = False,
    return_labels: bool = False,
    return_sizes: bool = False,
    dtype: type = None,
) -> Union[np.ndarray, NamedTuple]:
    """
    Fast version of `skimage.measure.label` which optionally returns number of connected components, labels and sizes.
    If 2 or more outputs are requested `NamedTuple` is returned.

    Parameters
    ----------
    label_image: np.ndarray
        image to label
    background: int
        consider all pixels with this value as background pixels, and label them as 0. By default, 0-valued pixels are
        considered as background pixels
    connectivity: int
        maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. Accepted values are ranging from 1
        to input.ndim. If None, a full connectivity of input.ndim is used
    return_num: bool
        whether to return the number of connected components
    return_labels: bool
        whether to return assigned labels
    return_sizes: bool
        whether to return sizes of connected components (excluding background)
    dtype:
        if specified, must be one of np.uint16, np.uint32 or np.uint64. If not specified, it will be automatically
        determined. Most of the time, you should leave this off so that the smallest safe dtype will be used. However,
        in some applications you can save an up-conversion in the next operation by outputting the appropriately sized
        type instead. Has no effect for python3.6

    Returns
    -------
    labeled_image: np.ndarray
        array of np.uint16, np.uint32 or np.uint64 numbers depending on the number of connected components and
        `dtype`
    num_components: int
        number of connected components excluding background. Returned if `return_num` is True
    labels: np.ndarray
        components labels. Returned if `return_labels` is True
    sizes: np.ndarray
        components sizes. Returned if `return_sizes` is True

    Examples
    --------
    >>> labeled = label(x)
    >>> labeled, num_components, sizes = label(x, return_num=True, return_sizes=True)
    >>> out = label(x, return_labels=True, return_sizes=True)
    >>> out.labeled_image, out.labels, out.sizes  # output fields can be accessed this way
    """
    ndim = label_image.ndim
    connectivity = connectivity or ndim

    if not 1 <= connectivity <= ndim:
        raise ValueError(f'Connectivity for {ndim}D image should be in [1, ..., {ndim}]. Got {connectivity}.')

    if ndim > 3:
        warn("Fast label is only supported for ndim<=3, Falling back to scikit-image's implementation.", stacklevel=2)
        labeled_image, num_components = skimage_label(
            label_image, background=background, return_num=True, connectivity=connectivity
        )
        if dtype is not None:
            labeled_image = labeled_image.astype(dtype, copy=False)
    else:
        if ndim == 1:
            label_image = label_image[None]

        if background:
            label_image = remap(
                label_image,
                {background: 0, 0: background},
                preserve_missing_labels=True,
                in_place=False,
            )

        labeled_image, num_components = connected_components(
            label_image,
            connectivity=_SKIMAGE2CC3D[(ndim, connectivity)],
            return_N=True,
            **{'out_dtype': dtype} if python_version()[:3] != '3.6' else {},
        )

        if ndim == 1:
            labeled_image = labeled_image[0]

    res = [('labeled_image', labeled_image)]

    if return_num:
        res.append(('num_components', num_components))
    if return_labels:
        res.append(('labels', np.array(range(1, num_components + 1))))
    if return_sizes:
        _, sizes = unique(labeled_image, return_counts=True)
        res.append(('sizes', sizes[1:] if 0 in labeled_image else sizes))

    if len(res) == 1:
        return labeled_image

    return namedtuple('Labeling', [subres[0] for subres in res])(*[subres[1] for subres in res])


def center_of_mass(
    array: np.ndarray,
    labels: np.ndarray = None,
    index: Union[int, Sequence[int]] = None,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> Union[Tuple[float, ...], List[Tuple[float, ...]]]:
    """
    Calculate the center of mass of the values.

    Works faster for ndim <= 3

    Parameters
    ----------
    array: np.ndarray
        data from which to calculate center-of-mass. The masses can either be positive or negative
    labels: np.ndarray
        labels for objects in input, as generated by `imops.measure.label`. Dimensions must be the same as input. If
        specified, `index` also must be specified and have same dtype
    index: Union[int, Sequence[int]]
        labels for which to calculate centers-of-mass. If specified, `labels` also must be specified and have same dtype
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used. If `labels` and `index` are specified, only 1 thread will be
        used
    backend: BackendLike
        which backend to use. `cython` and `scipy` are available, `cython` is used by default

    Returns
    -------
    center_of_mass: tuple, or list of tuples
        coordinates of centers-of-mass

    Examples
    --------
    >>> center = center_of_mass(np.ones((2, 2)))  # (0.5, 0.5)
    """
    if (labels is None) ^ (index is None):
        raise ValueError('`labels` and `index` should be both specified or both not specified.')

    backend = resolve_backend(backend, warn_stacklevel=3)

    if backend.name not in ('Scipy', 'Cython'):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    if backend.name == 'Scipy':
        return scipy_center_of_mass(array, labels, index)

    ndim = array.ndim
    if ndim > 3:
        warn("Fast center-of-mass is only supported for ndim<=3. Falling back to scipy's implementation.", stacklevel=2)
        return scipy_center_of_mass(array, labels, index)

    if labels is None:
        src_center_of_mass = _fast_center_of_mass if backend.fast else _center_of_mass
    else:
        is_sequence = isinstance(index, (Sequence, np.ndarray))
        index = np.array([index] if not is_sequence else index)

        if labels.shape != array.shape:
            raise ValueError(f'`array` and `labels` must be the same shape, got {array.shape} and {labels.shape}.')

        if labels.dtype != index.dtype:
            raise ValueError(f'`labels` and `index` must have same dtype, got {labels.dtype} and {index.dtype}.')

        if len(index) != len(unique(index.astype(int, copy=False))):
            raise ValueError('`index` should consist of unique values.')

        if num_threads > 1:
            warn('Using single-threaded implementation as `labels` and `index` are specified.', stacklevel=2)

        src_center_of_mass = _fast_labeled_center_of_mass if backend.fast else _labeled_center_of_mass

    if array.dtype != 'float64':
        array = array.astype(float)

    n_dummy = 3 - ndim
    if n_dummy:
        array = array[(None,) * n_dummy]

    if labels is None:
        return tuple(src_center_of_mass(array, num_threads))[n_dummy:]

    output = [tuple(x)[n_dummy:] for x in src_center_of_mass(array, labels[(None,) * n_dummy], index)]

    return output if is_sequence else output[0]
