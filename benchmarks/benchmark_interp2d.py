from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from imops.interp2d import Linear2DInterpolator

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg, load_npy_gz


class Interp2dSuite:
    params = [('C++', 'Scipy'), ('float32', 'float64'), NUMS_THREADS_TO_BENCHMARK]
    param_names = ['backend', 'dtype', 'num_threads']

    @discard_arg(1)
    @discard_arg(-1)
    def setup(self, dtype):
        x = load_npy_gz(Path(__file__).parent / 'tests' / 'test_data' / 'arr_0.npy.gz').astype(dtype)
        self.int_points = np.transpose((np.isnan(x)).nonzero())
        self.x_points = np.transpose((~np.isnan(x)).nonzero())
        self.x_values = x[~np.isnan(x)]

    @discard_arg(2)
    def time_interp2d(self, backend, num_threads):
        if backend == 'C++':
            Linear2DInterpolator(self.x_points, self.x_values, num_threads=num_threads)(self.int_points, fill_value=0.0)
        elif backend == 'Scipy':
            LinearNDInterpolator(self.x_points, self.x_values)(self.int_points, fill_value=0.0)
        else:
            raise NotImplementedError

    @discard_arg(2)
    def peakmem_interp2d(self, backend, num_threads):
        if backend == 'C++':
            Linear2DInterpolator(self.x_points, self.x_values, num_threads=num_threads)(self.int_points, fill_value=0.0)
        elif backend == 'Scipy':
            LinearNDInterpolator(self.x_points, self.x_values)(self.int_points, fill_value=0.0)
        else:
            raise NotImplementedError
