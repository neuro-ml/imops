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

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg


class MorphologySuite:
    params = [morphology_configs, ('bool', 'int64'), NUMS_THREADS_TO_BENCHMARK]
    param_names = ['backend', 'dtype', 'num_threads']

    @discard_arg(1)
    @discard_arg(-1)
    def setup(self, dtype):
        self.image = np.random.randint(0, 5 if dtype == 'int64' else 2, (256, 256, 256)).astype(dtype)

    # FIXME: generalize this code somehow
    @discard_arg(2)
    def time_closing(self, backend, num_threads):
        binary_closing(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_dilation(self, backend, num_threads):
        binary_dilation(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_erosion(self, backend, num_threads):
        binary_erosion(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_opening(self, backend, num_threads):
        binary_opening(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_closing(self, backend, num_threads):
        binary_closing(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_dilation(self, backend, num_threads):
        binary_dilation(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_erosion(self, backend, num_threads):
        binary_erosion(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_opening(self, backend, num_threads):
        binary_opening(self.image, num_threads=num_threads, backend=backend)
