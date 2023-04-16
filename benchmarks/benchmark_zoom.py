from itertools import product

import numpy as np


# TODO: Remove this crutch as soon as _configs.py appears in the master
try:
    from imops._configs import zoom_configs
except ModuleNotFoundError:
    from imops.backend import Cython, Numba, Scipy

    zoom_configs = [
        Scipy(),
        *[Cython(fast) for fast in [False, True]],
        *[Numba(*flags) for flags in product([False, True], repeat=3)],
    ]

from imops.zoom import zoom

from .common import discard_arg


class ZoomSuite:
    params = [zoom_configs, ('float32', 'float64'), [1, 2, 3, 4]]
    param_names = ['backend', 'dtype', 'ndim']

    @discard_arg(1)
    def setup(self, dtype, ndim):
        self.image = np.random.randn(2**24).astype(dtype).reshape([2 ** (24 // ndim) for _ in range(ndim)])

    @discard_arg(-1)
    @discard_arg(-1)
    def time_zoom(self, backend):
        zoom(self.image, 2, backend=backend)

    @discard_arg(-1)
    @discard_arg(-1)
    def peakmem_zoom(self, backend):
        zoom(self.image, 2, backend=backend)
