from itertools import product

import numpy as np


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

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg


class ZoomSuite:
    params = [[0, 1], NUMS_THREADS_TO_BENCHMARK, zoom_configs, ('float32', 'float64'), [1, 2, 3, 4]]
    param_names = ['order', 'num_threads', 'backend', 'dtype', 'ndim']

    @discard_arg(1)
    @discard_arg(1)
    @discard_arg(1)
    def setup(self, dtype, ndim):
        self.image = np.random.randn(2**24).astype(dtype).reshape([2 ** (24 // ndim) for _ in range(ndim)])

    @discard_arg(-1)
    @discard_arg(-1)
    def time_zoom(self, order, num_threads, backend):
        zoom(self.image, 2, order=order, num_threads=num_threads, backend=backend)

    @discard_arg(-1)
    @discard_arg(-1)
    def peakmem_zoom(self, order, num_threads, backend):
        zoom(self.image, 2, order=order, num_threads=num_threads, backend=backend)
