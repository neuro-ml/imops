import numpy as np


try:
    from imops._configs import rotate_configs
except ModuleNotFoundError:
    from imops.backend import Cython, Scipy

    rotate_configs = [
        Scipy(),
        *[Cython(fast) for fast in [False, True]],
    ]

from imops.rotate import rotate

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg


class RotateSuite:
    params = [[0, 1], NUMS_THREADS_TO_BENCHMARK, rotate_configs, ('float32', 'uint8', 'int16')]
    param_names = ['order', 'num_threads', 'backend', 'dtype']

    @discard_arg(1)
    @discard_arg(1)
    @discard_arg(1)
    def setup(self, dtype):
        self.image = (np.random.randn(256, 256, 256) * 100).astype(dtype)

    @discard_arg(-1)
    def time_rotate(self, order, num_threads, backend):
        rotate(self.image, 30, axes=(1, 2), order=order, num_threads=num_threads, backend=backend)

    @discard_arg(-1)
    def peakmem_rotate(self, order, num_threads, backend):
        rotate(self.image, 30, axes=(1, 2), order=order, num_threads=num_threads, backend=backend)
