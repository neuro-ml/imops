from collections import namedtuple
from platform import python_version
from typing import NamedTuple, Tuple, Union
from warnings import warn

import numpy as np
from cc3d import connected_components
from fastremap import remap, unique
from skimage.measure import label as skimage_label

from ._numeric import _mul, _sum
from .backend import BackendLike


# (ndim, skimage_connectivity) -> cc3d_connectivity
skimage2cc3d = {
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
        warn("Fast label is only supported for ndim<=3, Falling back to scikit-image's implementation.")
        labeled_image, num_components = skimage_label(
            label_image, background=background, return_num=True, connectivity=connectivity
        )
        if dtype is not None:
            labeled_image = labeled_image.astype(dtype)
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
            connectivity=skimage2cc3d[(ndim, connectivity)],
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
        res.append(('sizes', sizes[1:]))

    if len(res) == 1:
        return labeled_image

    return namedtuple('Labeling', [subres[0] for subres in res])(*[subres[1] for subres in res])


def center_of_mass(array: np.ndarray, num_threads: int = -1, backend: BackendLike = None) -> Tuple[float, ...]:
    """
    Calculate the center of mass of the values

    Parameters
    ----------
    array: np.ndarray
        data from which to calculate center-of-mass. The masses can either be positive or negative
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. Only `cython` is available

    Returns
    -------
    center_of_mass: tuple, or list of tuples
        coordinates of centers-of-mass

    Examples
    --------
    >>> center = center_of_mass(np.ones((2, 2)))  # (0.5, 0.5)
    """
    if array.dtype == bool:
        array = array.astype(float)

    normalizer = _sum(array.ravel(), num_threads=num_threads, backend=backend)
    grids = np.ogrid[list(map(slice, array.shape))]

    return tuple(
        _sum(
            _mul(
                array,
                grids[dim].astype(array.dtype),
                num_threads=num_threads,
                backend=backend,
            ).ravel(),
            num_threads=num_threads,
            backend=backend,
        )
        / normalizer
        for dim in range(array.ndim)
    )
