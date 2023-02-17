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


class Interp1dSuite:
    params = interp1d_configs

    def setup(self, backend):
        self.image = np.random.randn(256, 256, 256)
        self.x = np.random.randn(256)
        self.x_new = np.random.randn(256)

    def time_interp1d(self, backend):
        interp1d(self.x, self.image, bounds_error=False, fill_value='extrapolate', backend=backend)(self.x_new)

    def peakmem_interp1d(self, backend):
        interp1d(self.x, self.image, bounds_error=False, fill_value='extrapolate', backend=backend)(self.x_new)
