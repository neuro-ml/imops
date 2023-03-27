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
    params = [zoom_configs, ('float32', 'float64')]
    param_names = ['backend', 'dtype']

    @discard_arg(1)
    def setup(self, dtype):
        self.image_1d = np.random.randn(2**24).astype(dtype)
        self.image_2d = np.random.randn(4096, 4096).astype(dtype)
        self.image_3d = np.random.randn(256, 256, 256).astype(dtype)
        self.image_4d = np.random.randn(64, 64, 64, 64).astype(dtype)

    @discard_arg(2)
    def time_zoom_1d(self, backend):
        zoom(self.image_1d, 2, backend=backend)

    @discard_arg(2)
    def time_zoom_2d(self, backend):
        zoom(self.image_2d, 2, backend=backend)

    @discard_arg(2)
    def time_zoom_3d(self, backend):
        zoom(self.image_3d, 2, backend=backend)

    @discard_arg(2)
    def time_zoom_4d(self, backend):
        zoom(self.image_4d, 2, backend=backend)

    @discard_arg(2)
    def peakmem_zoom_1d(self, backend):
        zoom(self.image_1d, 2, backend=backend)

    @discard_arg(2)
    def peakmem_zoom_2d(self, backend):
        zoom(self.image_2d, 2, backend=backend)

    @discard_arg(2)
    def peakmem_zoom_3d(self, backend):
        zoom(self.image_3d, 2, backend=backend)

    @discard_arg(2)
    def peakmem_zoom_4d(self, backend):
        zoom(self.image_4d, 2, backend=backend)
