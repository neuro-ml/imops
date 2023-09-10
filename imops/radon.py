from typing import Sequence, Tuple, Union

import numpy as np
from scipy.fftpack import fft, ifft

from .backend import BackendLike, resolve_backend
from .numeric import copy
from .src._backprojection import backprojection3d
from .src._fast_backprojection import backprojection3d as fast_backprojection3d
from .src._fast_radon import radon3d as fast_radon3d
from .src._radon import radon3d
from .utils import normalize_num_threads


def radon(
    image: np.ndarray,
    axes: Tuple[int, int] = None,
    theta: Union[int, Sequence[float]] = 180,
    return_fill: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Fast implementation of Radon transform. Adapted from scikit-image.

    Parameters
    ----------
    image: np.ndarray
        an n-dimensional array with at least 2 axes
    axes: tuple[int, int]
        the axes in the `image` along which the Radon transform will be applied.
        The `image` shape along the `axes` must be of the same length
    theta: int | Sequence[float]
        the angles for which the Radon transform will be computed. If it is an integer - the angles will
        be evenly distributed between 0 and 180, `theta` values in total
    return_fill: bool
        whether to return the value that fills the image outside the circle working area
    num_threads: int
        the number of threads to be used for parallel computation. By default - equals to the number of cpu cores
    backend: str | Backend
        the execution backend. Currently only "Cython" is avaliable

    Returns
    -------
    sinogram: np.ndarray
        the result of the Radon transform
    fill_value: float
        the value that fills the image outside the circle working area. Returned only if `return_fill` is True

    Examples
    --------
    >>> sinogram = radon(image)  # 2d image
    >>> sinogram, fill_value = radon(image, return_fill=True)  # 2d image with fill value
    >>> sinogram = radon(image, axes=(-2, -1))  # nd image
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Cython',):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    image, axes, extra = normalize_axes(image, axes)
    if image.shape[1] != image.shape[2]:
        raise ValueError(
            f'The image must be square along the provided axes ({axes}), but has shape: {image.shape[1:]}.'
        )

    if isinstance(theta, int):
        theta = np.linspace(0, 180, theta, endpoint=False)

    size = image.shape[1]
    radius = size // 2
    xs = np.arange(-radius, size - radius)
    squared = xs**2
    outside_circle = (squared[:, None] + squared[None, :]) > radius**2
    values = image[:, outside_circle]
    min_, max_ = values.min(), values.max()
    if max_ - min_ > 0.1:
        raise ValueError(
            f'The image must be constant outside the circle. ' f'Got values ranging from {min_} to {max_}.'
        )

    if min_ != 0 or max_ != 0:
        # FIXME: how to accurately pass `num_threads` and `backend` arguments to `copy`?
        image = copy(image, order='C')
        image[:, outside_circle] = 0

    # TODO: f(arange)?
    limits = ((squared[:, None] + squared[None, :]) > (radius + 2) ** 2).sum(0) // 2

    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    radon3d_ = fast_radon3d if backend.fast else radon3d

    sinogram = radon3d_(image, np.deg2rad(theta, dtype=image.dtype), limits, num_threads)

    result = restore_axes(sinogram, axes, extra)
    if return_fill:
        result = result, min_

    return result


def inverse_radon(
    sinogram: np.ndarray,
    axes: Tuple[int, int] = None,
    theta: Union[int, Sequence[float]] = None,
    fill_value: float = 0,
    a: float = 0,
    b: float = 1,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast implementation of inverse Radon transform. Adapted from scikit-image.

    Parameters
    ----------
    sinogram: np.ndarray
        an n-dimensional array with at least 2 axes
    axes: tuple[int, int]
        the axes in the `image` along which the inverse Radon transform will be applied
    theta: int | Sequence[float]
        the angles for which the inverse Radon transform will be computed. If it is an integer - the angles will
        be evenly distributed between 0 and 180, `theta` values in total
    fill_value: float
        the value that fills the image outside the circle working area. Can be returned by `radon`
    a: float
        the first parameter of the sharpen filter
    b: float
        the second parameter of the sharpen filter
    num_threads: int
        the number of threads to be used for parallel computation. By default - equals to the number of cpu cores
    backend: str | Backend
        the execution backend. Currently only "Cython" is avaliable

    Returns
    -------
    image: np.ndarray
        the result of the inverse Radon transform

    Examples
    --------
    >>> image = inverse_radon(sinogram)  # 2d image
    >>> image = inverse_radon(sinogram, fill_value=-1000)  # 2d image with fill value
    >>> image = inverse_radon(sinogram, axes=(-2, -1))  # nd image
    """
    backend = resolve_backend(backend, warn_stacklevel=3)
    if backend.name not in ('Cython',):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    sinogram, axes, extra = normalize_axes(sinogram, axes)

    if theta is None:
        theta = sinogram.shape[-1]
    if isinstance(theta, int):
        theta = np.linspace(0, 180, theta, endpoint=False)

    angles_count = len(theta)
    if angles_count != sinogram.shape[-1]:
        raise ValueError(
            f'The given `theta` (size {angles_count}) does not match the number of '
            f'projections in `sinogram` ({sinogram.shape[-1]}).'
        )
    output_size = sinogram.shape[1]
    sinogram = _sinogram_circle_to_square(sinogram)

    img_shape = sinogram.shape[1]
    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    # TODO: why *2?
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, 0), (0, projection_size_padded - img_shape), (0, 0))
    padded_sinogram = np.pad(sinogram, pad_width, mode='constant', constant_values=0)
    fourier_filter = _smooth_sharpen_filter(projection_size_padded, a, b)

    # Apply filter in Fourier domain
    fourier_img = fft(padded_sinogram, axis=1) * fourier_filter
    filtered_sinogram = np.real(ifft(fourier_img, axis=1)[:, :img_shape, :])

    radius = output_size // 2
    xs = np.arange(-radius, output_size - radius)
    squared = xs**2
    inside_circle = (squared[:, None] + squared[None, :]) <= radius**2

    dtype = sinogram.dtype
    filtered_sinogram = filtered_sinogram.astype(dtype, copy=False)
    theta, xs = np.deg2rad(theta, dtype=dtype), xs.astype(dtype, copy=False)

    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)

    backprojection3d_ = fast_backprojection3d if backend.fast else backprojection3d

    reconstructed = np.asarray(
        backprojection3d_(filtered_sinogram, theta, xs, inside_circle, fill_value, img_shape, output_size, num_threads)
    )

    return restore_axes(reconstructed, axes, extra)


def normalize_axes(x: np.ndarray, axes):
    if x.ndim < 2:
        raise ValueError(f'Radon transform requires an array with at least 2 dimensions. {x.ndim}-dim array provided')
    if axes is None:
        if x.ndim > 2:
            raise ValueError('For arrays of higher dimensionality the `axis` arguments is required')
        axes = [0, 1]

    axes = np.core.numeric.normalize_axis_tuple(axes, x.ndim, 'axes')
    x = np.moveaxis(x, axes, (-2, -1))
    extra = x.shape[:-2]
    x = x.reshape(-1, *x.shape[-2:])
    return x, axes, extra


def restore_axes(x: np.ndarray, axes: tuple, extra: tuple) -> np.ndarray:
    x = x.reshape(*extra, *x.shape[-2:])
    x = np.moveaxis(x, (-2, -1), axes)
    return x


def _ramp_filter(size: int) -> np.ndarray:
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(fft(f))

    return fourier_filter.reshape(-1, 1)


def _smooth_sharpen_filter(size: int, a: float, b: float) -> np.ndarray:
    ramp = _ramp_filter(size)
    return ramp * (1 + a * (ramp**b))


def _sinogram_circle_to_square(sinogram: np.ndarray) -> np.ndarray:
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[1]))
    pad = diagonal - sinogram.shape[1]
    old_center = sinogram.shape[1] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((0, 0), (pad_before, pad - pad_before), (0, 0))

    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)
