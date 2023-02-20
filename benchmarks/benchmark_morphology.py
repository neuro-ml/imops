import numpy as np


# TODO: Remove this crutch as soon as _configs.py appears in the master
try:
    from imops._configs import morphology_configs
except ModuleNotFoundError:
    from imops.backend import Cython, Scipy

    morphology_configs = [
        Scipy(),
        *[Cython(fast) for fast in [False, True]],
    ]

from imops.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening

from .common import discard_arg


class MorphologySuite:
    params = [morphology_configs, ('bool', 'int64')]
    param_names = ['backend', 'dtype']

    @discard_arg(1)
    def setup(self, dtype):
        self.image = np.random.randint(0, 5 if dtype is int else 2, (256, 256, 256)).astype(dtype)

    @discard_arg(2)
    def time_closing(self, backend):
        binary_closing(self.image, backend=backend)

    @discard_arg(2)
    def time_dilation(self, backend):
        binary_dilation(self.image, backend=backend)

    @discard_arg(2)
    def time_erosion(self, backend):
        binary_erosion(self.image, backend=backend)

    @discard_arg(2)
    def time_opening(self, backend):
        binary_opening(self.image, backend=backend)

    @discard_arg(2)
    def peakmem_closing(self, backend):
        binary_closing(self.image, backend=backend)

    @discard_arg(2)
    def peakmem_dilation(self, backend):
        binary_dilation(self.image, backend=backend)

    @discard_arg(2)
    def peakmem_erosion(self, backend):
        binary_erosion(self.image, backend=backend)

    @discard_arg(2)
    def peakmem_opening(self, backend):
        binary_opening(self.image, backend=backend)
