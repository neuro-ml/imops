from typing import Union
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

from .backend import BackendLike, resolve_backend
from .src._fast_zoom import _interp1d as cython_fast_interp1d
from .src._zoom import _interp1d as cython_interp1d
from .utils import FAST_MATH_WARNING, normalize_num_threads


class interp1d:
    """
    Faster parallelizable version of `scipy.interpolate.interp1d` for fp32 / fp64 inputs.

    Works faster only for ndim <= 3. Shares interface with `scipy.interpolate.interp1d` except for
    - `num_threads` argument defining how many threads to use (all available threads are used by default)
    - `backend` argument defining which backend to use. `numba`, `cython` and `scipy` are available,
        `cython` is used by default.

    See `https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html`
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
        backend = resolve_backend(backend)
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
            )
            self.scipy_interp1d = scipy_interp1d(x, y, kind, axis, copy, bounds_error, fill_value, assume_sorted)
        else:
            if bounds_error and fill_value == 'extrapolate':
                raise ValueError('Cannot extrapolate and raise at the same time.')

            if fill_value == 'extrapolate' and len(x) < 2 or len(y) < 2:
                raise ValueError('x and y arrays must have at least 2 entries.')

            if fill_value == 'extrapolate':
                self.bounds_error = False
            else:
                self.bounds_error = True if bounds_error is None else bounds_error

            if len(x) != y.shape[axis]:
                raise ValueError('x and y arrays must be equal in length along interpolation axis.')

            self.axis = axis

            if axis not in (-1, y.ndim - 1):
                y = np.swapaxes(y, -1, axis)

            self.fill_value = fill_value
            self.scipy_interp1d = None
            self.x = np.copy(x) if copy else x
            self.n_dummy = 3 - y.ndim
            self.y = y[(None,) * self.n_dummy] if self.n_dummy else y

            if copy:
                self.y = np.copy(self.y)

            self.assume_sorted = assume_sorted

            if backend.name == 'Cython':
                if backend.fast:
                    warn(FAST_MATH_WARNING)
                    self.src_interp1d = cython_fast_interp1d
                else:
                    self.src_interp1d = cython_interp1d

            if backend.name == 'Numba':
                from numba import njit

                from .src._numba_zoom import _interp1d as numba_interp1d

                njit_kwargs = {kwarg: getattr(backend, kwarg) for kwarg in backend.__dataclass_fields__.keys()}
                self.src_interp1d = njit(**njit_kwargs)(numba_interp1d)

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        num_threads = normalize_num_threads(self.num_threads, self.backend)

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
            self.x.astype(np.float64),
            x_new.astype(np.float64),
            self.bounds_error,
            0.0 if extrapolate else self.fill_value,
            extrapolate,
            self.assume_sorted,
            *args,
        )

        if self.backend.name == 'Numba':
            set_num_threads(old_num_threads)

        out = out.astype(max(self.y.dtype, self.x.dtype, x_new.dtype, key=lambda x: x.type(0).itemsize))

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
