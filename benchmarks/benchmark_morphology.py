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


class MorphologySuite:
    params = morphology_configs

    def setup(self, backend):
        self.image = np.random.binomial(1, 0.5, (256, 256, 256)).astype(bool)

    def time_closing(self, backend):
        binary_closing(self.image)

    def time_dilation(self, backend):
        binary_dilation(self.image)

    def time_erosion(self, backend):
        binary_erosion(self.image)

    def time_opening(self, backend):
        binary_opening(self.image)

    def peakmem_closing(self, backend):
        binary_closing(self.image)

    def peakmem_dilation(self, backend):
        binary_dilation(self.image)

    def peakmem_erosion(self, backend):
        binary_erosion(self.image)

    def peakmem_opening(self, backend):
        binary_opening(self.image)
