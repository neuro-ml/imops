from typing import Union
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

from .backend import BackendLike, resolve_backend
from .numeric import copy as _copy
from .src._fast_zoom import _interp1d as cython_fast_interp1d
from .src._zoom import _interp1d as cython_interp1d
from .utils import normalize_num_threads


class interp1d:
    """
    Faster parallelizable version of `scipy.interpolate.interp1d` for fp32 / fp64 inputs.

    Works faster only for ndim <= 3. Shares interface with `scipy.interpolate.interp1d` except for `num_threads` and
    `backend` arguments.

    Parameters
    ----------
    x: np.ndarray
        1-dimensional array of real values (aka coordinates)
    y: np.ndarray
        n-dimensional array of real values. The length of y along the interpolation axis must be equal to the x length
    kind: Union[int, str]
        specifies the kind of interpolation as a string or as an integer specifying the order of interpolation to use.
        Only kind=1 and 'linear` are fast and parallelizable, other kinds will force to use `scipy` implementation
    axis: int
        specifies the axis of y along which to interpolate. Interpolation defaults to the last axis of y
    copy: bool
        if True, the class makes internal copies of x and y. If False, references to x and y are used
    bounds_error: bool
        if True, a ValueError is raised any time interpolation is attempted on a value outside of the range of x where
        extrapolation is necessary. If False, out of bounds values are assigned fill_value. By default, an error is
        raised unless fill_value='extrapolate'
    fill_value: Union[float, str]
        if a float, this value will be used to fill in for requested points outside of the data range. If not provided,
        then the default is NaN. If 'extrapolate', values for points outside of the data range will be extrapolated
    assume_sorted: bool
        if False, values of x can be in any order and they are sorted first. If True, x has to be an array of
        monotonically increasing values
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    backend: BackendLike
        which backend to use. `numba`, `cython` and `scipy` are available, `cython` is used by default

    Methods
    -------
    __call__

    Examples
    --------
    >>> import numpy as np
    >>> from imops.interp1d import interp1d
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interp1d(x, y)
    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        kind: Union[int, str] = 'linear',
        axis: int = -1,
        copy: bool = True,
        bounds_error: bool = None,
        fill_value: Union[float, str] = np.nan,
        assume_sorted: bool = False,
        num_threads: int = -1,
        backend: BackendLike = None,
    ) -> None:
        backend = resolve_backend(backend, warn_stacklevel=3)
        if backend.name not in ('Scipy', 'Numba', 'Cython'):
            raise ValueError(f'Unsupported backend "{backend.name}".')

        self.backend = backend
        self.dtype = y.dtype
        self.num_threads = num_threads

        if backend.name == 'Scipy':
            self.scipy_interp1d = scipy_interp1d(x, y, kind, axis, copy, bounds_error, fill_value, assume_sorted)
        elif self.dtype not in (np.float32, np.float64) or y.ndim > 3 or kind not in ('linear', 1):
            warn(
                "Fast interpolation is only supported for ndim<=3, dtype=float32 or float64, order=1 or 'linear'. "
                "Falling back to scipy's implementation.",
                stacklevel=2,
            )
            self.scipy_interp1d = scipy_interp1d(x, y, kind, axis, copy, bounds_error, fill_value, assume_sorted)
        else:
            if len(x) != y.shape[axis]:
                raise ValueError(
                    f'x and y arrays must be equal in length along interpolation axis: {len(x)} vs {y.shape[axis]}.'
                )

            if bounds_error and fill_value == 'extrapolate':
                raise ValueError('Cannot extrapolate and raise at the same time.')

            if fill_value == 'extrapolate' and len(x) < 2 or y.shape[axis] < 2:
                raise ValueError('x and y arrays must have at least 2 entries.')

            if fill_value == 'extrapolate':
                self.bounds_error = False
            else:
                self.bounds_error = True if bounds_error is None else bounds_error

            self.axis = axis

            if axis not in (-1, y.ndim - 1):
                y = np.swapaxes(y, -1, axis)

            self.fill_value = fill_value
            self.scipy_interp1d = None
            # FIXME: how to accurately pass `num_threads` and `backend` arguments to `copy`?
            self.x = _copy(x, order='C') if copy else x
            self.n_dummy = 3 - y.ndim
            self.y = y[(None,) * self.n_dummy] if self.n_dummy else y

            if copy:
                self.y = _copy(self.y, order='C')

            self.assume_sorted = assume_sorted

            if backend.name == 'Cython':
                self.src_interp1d = cython_fast_interp1d if backend.fast else cython_interp1d

            if backend.name == 'Numba':
                from numba import njit

                from .src._numba_zoom import _interp1d as numba_interp1d

                njit_kwargs = {kwarg: getattr(backend, kwarg) for kwarg in backend.__dataclass_fields__.keys()}
                self.src_interp1d = njit(**njit_kwargs)(numba_interp1d)

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the interpolant

        Parameters
        ----------
        x_new: np.ndarray
            1d array points to evaluate the interpolant at.

        Returns
        -------
        y_new: np.ndarray
            interpolated values. Shape is determined by replacing the interpolation axis in the original array with
            the shape of x
        """
        num_threads = normalize_num_threads(self.num_threads, self.backend, warn_stacklevel=3)

        if self.scipy_interp1d is not None:
            return self.scipy_interp1d(x_new)

        extrapolate = self.fill_value == 'extrapolate'
        args = () if self.backend.name in ('Numba',) else (num_threads,)

        if self.backend.name == 'Numba':
            from numba import get_num_threads, set_num_threads

            old_num_threads = get_num_threads()
            set_num_threads(num_threads)
        # TODO: Figure out how to properly handle multiple type signatures in Cython and remove `.astype`-s
        out = self.src_interp1d(
            self.y,
            self.x.astype(np.float64, copy=False),
            x_new.astype(np.float64, copy=False),
            self.bounds_error,
            0.0 if extrapolate else self.fill_value,
            extrapolate,
            self.assume_sorted,
            *args,
        )

        if self.backend.name == 'Numba':
            set_num_threads(old_num_threads)

        out = out.astype(max(self.y.dtype, self.x.dtype, x_new.dtype, key=lambda x: x.type(0).itemsize), copy=False)

        if self.n_dummy:
            out = out[(0,) * self.n_dummy]
        if self.axis not in (-1, out.ndim - 1):
            out = np.swapaxes(out, -1, self.axis)
        # FIXME: fix behaviour with np.inf-s
        if np.isnan(out).any():
            if not np.isinf(out).any():
                raise RuntimeError("Can't decide how to handle nans in the output.")

            have_neg = np.isneginf(out).any()
            have_pos = np.isposinf(out).any()

            if have_pos and have_neg:
                raise RuntimeError("Can't decide how to handle nans in the output.")

            if have_pos:
                return np.nan_to_num(out, copy=False, nan=np.inf, posinf=np.inf)

            return np.nan_to_num(out, copy=False, nan=-np.inf, neginf=-np.inf)

        return out
