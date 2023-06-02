from itertools import product

import numpy as np


# TODO: Remove this crutch as soon as _configs.py appears in the master
try:
    from imops._configs import interp1d_configs
except ModuleNotFoundError:
    from imops.backend import Cython, Numba, Scipy

    interp1d_configs = [
        Scipy(),
        *[Cython(fast) for fast in [False, True]],
        *[Numba(*flags) for flags in product([False, True], repeat=3)],
    ]

from imops.interp1d import interp1d

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg


class Interp1dSuite:
    params = [interp1d_configs, ('float32', 'float64'), NUMS_THREADS_TO_BENCHMARK]
    param_names = ['backend', 'dtype', 'num_threads']

    @discard_arg(1)
    @discard_arg(-1)
    def setup(self, dtype):
        self.image = np.random.randn(256, 256, 256).astype(dtype)
        self.x = np.random.randn(256).astype(dtype)
        self.x_new = np.random.randn(256).astype(dtype)

    @discard_arg(2)
    def time_interp1d(self, backend, num_threads):
        interp1d(
            self.x, self.image, bounds_error=False, fill_value='extrapolate', num_threads=num_threads, backend=backend
        )(self.x_new)

    @discard_arg(2)
    def peakmem_interp1d(self, backend, num_threads):
        interp1d(
            self.x, self.image, bounds_error=False, fill_value='extrapolate', num_threads=num_threads, backend=backend
        )(self.x_new)
