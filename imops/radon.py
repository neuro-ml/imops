from typing import Sequence, Tuple, Union
from warnings import warn

import numpy as np
from scipy.fftpack import fft, ifft

from .backend import BackendLike, resolve_backend
from .src._backprojection import backprojection3d
from .src._fast_backprojection import backprojection3d as fast_backprojection3d
from .src._fast_radon import radon3d as fast_radon3d
from .src._radon import radon3d
from .utils import FAST_MATH_WARNING, normalize_axes, normalize_num_threads, restore_axes


def radon(
    image: np.ndarray,
    axes: Tuple[int, int] = None,
    theta: Union[int, Sequence[float]] = None,
    return_fill: bool = False,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast implementation of Radon transform. Adapted from scikit-image.
    """
    backend = resolve_backend(backend)
    if backend.name not in ('Cython',):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    image, axes, squeeze = normalize_axes(image, axes)
    if image.shape[1] != image.shape[2]:
        raise ValueError(f'The image must be square along the provided axes. Shape: {image.shape[1:]}.')

    if theta is None:
        theta = 180
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
        image = image.copy()
        image[:, outside_circle] = 0

    # TODO: f(arange)?
    limits = ((squared[:, None] + squared[None, :]) > (radius + 2) ** 2).sum(0) // 2

    num_threads = normalize_num_threads(num_threads, backend)

    if backend.fast:
        warn(FAST_MATH_WARNING)
        radon3d_ = fast_radon3d
    else:
        radon3d_ = radon3d

    sinogram = radon3d_(image, np.deg2rad(theta), limits, num_threads)

    result = restore_axes(sinogram, axes, squeeze)
    if return_fill:
        result = result, min_

    return result


def inverse_radon(
    sinogram: np.ndarray,
    a: float = 0,
    b: float = 1,
    fill_value: float = 0,
    theta: Union[int, Sequence[float]] = None,
    axes: Tuple[int, int] = None,
    num_threads: int = -1,
    backend: BackendLike = None,
) -> np.ndarray:
    """
    Fast implementation of inverse Radon transform. Adapted from scikit-image.
    """
    backend = resolve_backend(backend)
    if backend.name not in ('Cython',):
        raise ValueError(f'Unsupported backend "{backend.name}".')

    sinogram, axes, squeeze = normalize_axes(sinogram, axes)

    if theta is None:
        theta = sinogram.shape[-1]
    if isinstance(theta, int):
        theta = np.linspace(0, 180, theta, endpoint=False)

    angles_count = len(theta)
    if angles_count != sinogram.shape[-1]:
        raise ValueError('The given `theta` does not match the number of projections in `sinogram`.')
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
    filtered_sinogram = filtered_sinogram.astype(dtype)
    theta, xs = np.deg2rad(theta).astype(dtype), xs.astype(dtype)

    num_threads = normalize_num_threads(num_threads, backend)

    if backend.fast:
        warn(FAST_MATH_WARNING)
        backprojection3d_ = fast_backprojection3d
    else:
        backprojection3d_ = backprojection3d

    reconstructed = backprojection3d_(
        filtered_sinogram, theta, xs, inside_circle, fill_value, img_shape, output_size, num_threads
    )

    return restore_axes(reconstructed, axes, squeeze)


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
