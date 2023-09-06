import numpy as np

from imops._configs import numeric_configs
from imops.numeric import _fill, copy, full, pointwise_add

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg


class NumericSuite:
    params = [numeric_configs, ('float16', 'float32', 'float64', 'int16', 'int32', 'int64'), NUMS_THREADS_TO_BENCHMARK]
    param_names = ['backend', 'dtype', 'num_threads']
    timeout = 300

    @discard_arg(1)
    @discard_arg(-1)
    def setup(self, dtype):
        self.shape = (512, 512, 512)
        self.empty_3d = np.empty(self.shape, dtype=dtype)
        self.nums1_3d = (32 * np.random.randn(512, 512, 512)).astype(dtype)
        self.nums2_3d = (32 * np.random.randn(512, 512, 512)).astype(dtype)

    @discard_arg(2)
    def time_add_array(self, backend, num_threads):
        pointwise_add(self.nums1_3d, self.nums2_3d, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_add_value(self, backend, num_threads):
        pointwise_add(self.nums1_3d, 1, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_copy(self, backend, num_threads):
        copy(self.nums1_3d, num_threads, backend)

    def time_full(self, backend, dtype, num_threads):
        full(self.shape, 42, dtype, num_threads, backend)

    @discard_arg(2)
    def time_fill(self, backend, dtype, num_threads):
        _fill(self.empty_3d, 42, num_threads, backend)

    @discard_arg(2)
    def peakmem_add_array(self, backend, num_threads):
        pointwise_add(self.nums1_3d, self.nums2_3d, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_add_value(self, backend, num_threads):
        pointwise_add(self.nums1_3d, 1, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_copy(self, backend, num_threads):
        copy(self.nums1_3d, num_threads, backend)

    def peakmem_full(self, backend, dtype, num_threads):
        full(self.shape, 42, dtype, num_threads, backend)

    @discard_arg(2)
    def peakmem_fill(self, backend, dtype, num_threads):
        _fill(self.empty_3d, 42, num_threads, backend)
