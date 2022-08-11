import os
from typing import Union

import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

from .src._fast_zoom import _interp1d


class interp1d:
    """
    Faster parallelizable version of `scipy.interpolate.interp1d`

    Works faster only for ndim <= 3. Shares interface with `scipy.interpolate.interp1d`
    except for `num_threads` argument defining how many threads to use, all available threads are used by default.

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
    ) -> None:
        if y.dtype not in (np.float32, np.float64):
            raise ValueError('Only fp32 and fp64 dtypes are allowed for interp1d.')
        if y.ndim > 3:
            self.scipy_interp1d = scipy_interp1d(x, y, kind, axis, copy, bounds_error, fill_value, assume_sorted)
        else:
            if kind not in ('linear', 1):
                raise NotImplementedError("Only kind 'linear' and 1 are implemented for ndim <= 3.")
            if bounds_error and fill_value == 'extrapolate':
                raise ValueError('Cannot extrapolate and raise at the same time.')
            if fill_value == 'extrapolate' and len(x) < 2 or len(y) < 2:
                raise ValueError('x and y arrays must have at least 2 entries.')

            if fill_value == 'extrapolate':
                self.bounds_error = False
            else:
                self.bounds_error = True if bounds_error is None else bounds_error

            self.fill_value = fill_value
            self.scipy_interp1d = None
            self.x = np.copy(x) if copy else x

            self.axis = axis
            if axis not in (-1, y.ndim - 1):
                y = np.swapaxes(y, -1, axis)

            self.n_dummy = 3 - y.ndim
            self.y = y[(None,) * self.n_dummy] if self.n_dummy else y
            if copy:
                self.y = np.copy(self.y)

            self.assume_sorted = assume_sorted
            self.num_threads = num_threads

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        if self.scipy_interp1d is not None:
            return self.scipy_interp1d(x_new)

        num_threads = self.num_threads if self.num_threads != -1 else os.cpu_count()
        extrapolate = self.fill_value == 'extrapolate'

        out = _interp1d(
            self.y,
            self.x,
            x_new,
            self.bounds_error,
            0.0 if extrapolate else self.fill_value,
            extrapolate,
            self.assume_sorted,
            num_threads,
        )

        if self.n_dummy:
            out = out[(0,) * self.n_dummy]

        if self.axis not in (-1, out.ndim - 1):
            out = np.swapaxes(out, -1, self.axis)

        # TODO: fix behaviour with np.inf-s
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
