from warnings import warn

import numpy as np
from scipy.ndimage import binary_dilation as scipy_binary_dilation, generate_binary_structure

from .backend import BackendLike, resolve_backend
from .src._fast_morphology import _binary_dilation as cython_fast_binary_dilation
from .src._morphology import _binary_dilation as cython_binary_dilation
from .utils import FAST_MATH_WARNING, normalize_num_threads


def binary_dilation(
    image: np.ndarray, footprint: np.ndarray = None, num_threads: int = -1, backend: BackendLike = None
):
    """
    Fast parallelizable binary morphological dilation of an image

    See `https://scikit-image.org/docs/stable/api/skimage.morphology#skimage.morphology.binary_dilation`

    Parameters
    ----------
    image
    footprint
    num_threads
        the number of threads to use for computation. Default = the cpu count.
    backend
        which backend to use. `cython` and `scipy` are available, `cython` is used by default.
    """
    backend = resolve_backend(backend)
    if backend.name not in ('Cython', 'Scipy'):
        raise ValueError(f'Unsupported backend "{backend.name}"')

    ndim = image.ndim
    num_threads = normalize_num_threads(num_threads, backend)

    if backend.name == 'Scipy':
        return scipy_binary_dilation(image, footprint)

    if ndim > 3:
        warn(
            "Fast binary dilation is only supported for ndim<=3. Falling back to scipy's implementation.",
        )
        scipy_binary_dilation(image, footprint)

    if backend.fast:
        warn(FAST_MATH_WARNING)
        src_binary_dilation = cython_fast_binary_dilation
    else:
        src_binary_dilation = cython_binary_dilation

    if footprint.ndim != ndim:
        raise ValueError('Input image and footprint number of dimensions must be the same.')

    n_dummy = 3 - ndim

    if footprint is None:
        footprint = generate_binary_structure(ndim, 1)

    if n_dummy:
        image = image[(None,) * n_dummy]
        footprint = footprint[(None,) * n_dummy]

    out = src_binary_dilation(image.astype(bool), footprint.astype(bool), num_threads)

    if n_dummy:
        out = out[(0,) * n_dummy]

    return out.astype(bool)
