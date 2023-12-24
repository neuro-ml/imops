from pathlib import Path

import numpy as np
import pytest
import scipy
from deli import load
from scipy.interpolate import griddata

from imops.interp2d import Linear2DInterpolator


np.random.seed(1337)


@pytest.fixture(params=range(1, 5))
def n_jobs(request):
    return request.param


def test_test_data(n_jobs):
    tests_root = Path(__file__).parent
    for i in range(2):
        x = load(tests_root / 'test_data' / f'arr_{i}.npy.gz')
        int_points = np.transpose((np.isnan(x)).nonzero())
        x_points: np.ndarray = np.transpose((~np.isnan(x)).nonzero())
        x_values = x[~np.isnan(x)]
        imops_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs)(int_points, fill_value=0.0)
        triangles = scipy.spatial.Delaunay(x_points).simplices
        delaunay_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs, triangles=triangles)(
            int_points, fill_value=0.0
        )
        scipy_values = griddata(x_points, x_values, int_points, method='linear', fill_value=0.0)

        delta_ds = np.abs(delaunay_values - scipy_values)
        # delta_di = np.abs(delaunay_values - imops_values)
        delta_si = np.abs(scipy_values - imops_values)

        assert delta_ds.max() <= 1e-10 and delta_si.max() <= 5, f'Failed with big case, arr_{i}'

    for i in range(2, 7):
        x = load(tests_root / 'test_data' / f'arr_{i}.npy.gz')
        x_values = load(tests_root / 'test_data' / f'val_{i}.npy.gz')
        x_points = np.transpose(x.nonzero())
        int_points = np.transpose((~x).nonzero())
        imops_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs)(int_points, fill_value=0.0)
        triangles = scipy.spatial.Delaunay(x_points).simplices
        delaunay_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs, triangles=triangles)(
            int_points, fill_value=0.0
        )
        scipy_values = griddata(x_points, x_values, int_points, method='linear', fill_value=0.0)

        delta_ds = np.abs(delaunay_values - scipy_values)
        delta_di = np.abs(delaunay_values - imops_values)
        delta_si = np.abs(scipy_values - imops_values)

        assert (
            delta_di.max() <= 1.5 and delta_ds.max() <= 1e-10 and delta_si.max() <= 1.5
        ), f'Failed with small case: arr_{i}'
