import numpy as np

from imops._configs import numeric_configs
from imops._numeric import _mul, _sum

from .common import discard_arg


class NumericSuite:
    params = [numeric_configs, ('float32', 'float64', 'int32', 'int64')]
    param_names = ['backend', 'dtype']
    timeout = 300

    @discard_arg(1)
    def setup(self, dtype):
        self.nums_1d = (32 * np.random.randn(10**9)).astype(dtype)
        self.nums1_3d = (32 * np.random.randn(512, 512, 512)).astype(dtype)
        self.nums2_3d = (32 * np.random.randn(512, 512, 512)).astype(dtype)

    @discard_arg(2)
    def time_sum(self, backend):
        _sum(self.nums_1d, backend=backend)

    @discard_arg(2)
    def peakmem_sum(self, backend):
        _sum(self.nums_1d, backend=backend)

    @discard_arg(2)
    def time_pointwise_mul(self, backend):
        _mul(self.nums1_3d, self.nums2_3d, backend=backend)

    @discard_arg(2)
    def peakmem_pointwise_mul(self, backend):
        _mul(self.nums1_3d, self.nums2_3d, backend=backend)
