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
        x = load_npy_gz(Path(__file__).parent / 'data' / 'ribs.npy.gz').astype(dtype)
        add_cols = x.shape[1] // 4
        distances = np.concatenate((x[..., -add_cols:], x, x[..., :add_cols]), axis=1)
        self.int_points = np.transpose((np.isnan(distances)).nonzero())
        self.x_points = np.transpose((~np.isnan(distances)).nonzero())
        self.x_values = distances[~np.isnan(distances)]

    @discard_arg(2)
    def time_interp2d(self, backend, num_threads):
        if backend == 'C++':
            Linear2DInterpolator(self.x_points, self.x_values, num_threads=num_threads)(self.int_points, fill_value=0.0)
        elif backend == 'Scipy':
            LinearNDInterpolator(self.x_points, self.x_values)(self.int_points)
        else:
            raise NotImplementedError

    @discard_arg(2)
    def peakmem_interp2d(self, backend, num_threads):
        if backend == 'C++':
            Linear2DInterpolator(self.x_points, self.x_values, num_threads=num_threads)(self.int_points, fill_value=0.0)
        elif backend == 'Scipy':
            LinearNDInterpolator(self.x_points, self.x_values)(self.int_points)
        else:
            raise NotImplementedError
