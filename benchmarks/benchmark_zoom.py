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


class ZoomSuite:
    params = zoom_configs

    def setup(self, backend):
        self.image = np.random.randn(256, 256, 256)

    def time_zoom(self, backend):
        zoom(self.image, 2, backend=backend)

    def peakmem_zoom(self, backend):
        zoom(self.image, 2, backend=backend)
