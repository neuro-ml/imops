from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.fft import irfft, rfft

from .backend import BackendLike, resolve_backend
from .compat import normalize_axis_tuple
from .numeric import copy
from .src._backprojection import backprojection3d, by_angle
from .src._fast_backprojection import backprojection3d as fast_backprojection3d, by_angle as fast_by_angle
from .src._fast_radon import radon3d as fast_radon3d
from .src._radon import radon3d
from .utils import normalize_num_threads


def radon(
    image: np.ndarray,
    axes: Optional[Tuple[int, int]] = None,
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
    ```python
    sinogram = radon(image)  # 2d image
    sinogram, fill_value = radon(image, return_fill=True)  # 2d image with fill value
    sinogram = radon(image, axes=(-2, -1))  # nd image
    ```
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
    axes: Optional[Tuple[int, int]] = None,
    theta: Union[int, Sequence[float], None] = None,
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
    ```python
    image = inverse_radon(sinogram)  # 2d image
    image = inverse_radon(sinogram, fill_value=-1000)  # 2d image with fill value
    image = inverse_radon(sinogram, axes=(-2, -1))  # nd image
    ```
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
    dtype = sinogram.dtype
    img_shape, offset = _square_detector(output_size)
    num_threads = normalize_num_threads(num_threads, backend, warn_stacklevel=3)
    by_angle_ = fast_by_angle if backend.fast else by_angle
    backprojection3d_ = fast_backprojection3d if backend.fast else backprojection3d

    # The backprojection reads every slice of one detector sample at once, so slices go last and stay there.
    rows = by_angle_(np.ascontiguousarray(sinogram), img_shape, offset, num_threads)

    # Shorter than twice the detector and the ramp filter wraps around. A power of two transforms fastest.
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    fourier_filter = _smooth_sharpen_filter(padded_size, a, b).astype(dtype)
    spectrum = rfft(rows, n=padded_size, axis=1, workers=num_threads)
    spectrum *= fourier_filter
    filtered = irfft(spectrum, padded_size, axis=1, workers=num_threads, overwrite_x=True)

    radius = output_size // 2
    xs = np.arange(-radius, output_size - radius)
    squared = xs**2
    inside_circle = (squared[:, None] + squared[None, :]) <= radius**2
    theta, xs = np.deg2rad(theta, dtype=dtype), xs.astype(dtype, copy=False)

    reconstructed = np.asarray(
        backprojection3d_(filtered, theta, xs, inside_circle, fill_value, img_shape, num_threads)
    )

    return restore_axes(reconstructed, axes, extra)


def normalize_axes(x: np.ndarray, axes):
    if x.ndim < 2:
        raise ValueError(f'Radon transform requires an array with at least 2 dimensions. {x.ndim}-dim array provided')
    if axes is None:
        if x.ndim > 2:
            raise ValueError('For arrays of higher dimensionality the `axis` arguments is required')
        axes = [0, 1]

    axes = normalize_axis_tuple(axes, x.ndim, 'axes')
    x = np.moveaxis(x, axes, (-2, -1))
    extra = x.shape[:-2]
    x = x.reshape(-1, *x.shape[-2:])
    return x, axes, extra


def restore_axes(x: np.ndarray, axes: tuple, extra: tuple) -> np.ndarray:
    x = x.reshape(*extra, *x.shape[-2:])
    x = np.moveaxis(x, (-2, -1), axes)
    return x


def _ramp_filter(size: int) -> np.ndarray:
    """The ramp filter over the non-negative frequencies, which is all a real signal needs."""
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    return 2 * np.real(rfft(f)).reshape(-1, 1)


def _smooth_sharpen_filter(size: int, a: float, b: float) -> np.ndarray:
    ramp = _ramp_filter(size)
    return ramp * (1 + a * (ramp**b))


def _square_detector(size: int) -> Tuple[int, int]:
    """The detector that holds the rotated circle, and where the old one starts inside it."""
    diagonal = int(np.ceil(np.sqrt(2) * size))

    return diagonal, diagonal // 2 - size // 2
