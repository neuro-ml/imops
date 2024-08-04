from pathlib import Path

import numpy as np
import pytest
import scipy
from deli import load
from scipy.interpolate import griddata

from imops.interp2d import Linear2DInterpolator
from imops.utils import make_immutable


tests_root = Path(__file__).parent
np.random.seed(1337)


@pytest.fixture(params=range(1, 5))
def num_threads(request):
    return request.param


@pytest.fixture
def example():
    x = load(tests_root / 'test_data' / 'arr_0.npy.gz')
    int_points = np.transpose((np.isnan(x)).nonzero())
    x_points = np.transpose((~np.isnan(x)).nonzero())

    return x_points, int_points


def test_test_data(num_threads):
    for i in range(2):
        x = load(tests_root / 'test_data' / f'arr_{i}.npy.gz')
        int_points = np.transpose((np.isnan(x)).nonzero())
        x_points: np.ndarray = np.transpose((~np.isnan(x)).nonzero())
        x_values = x[~np.isnan(x)]

        if np.random.binomial(1, 0.5):
            make_immutable(x_points)
        if np.random.binomial(1, 0.5):
            make_immutable(x_values)
        if np.random.binomial(1, 0.5):
            make_immutable(int_points)

        imops_values = Linear2DInterpolator(x_points, x_values, num_threads=num_threads)(int_points, fill_value=0.0)
        triangles = scipy.spatial.Delaunay(x_points).simplices
        delaunay_values = Linear2DInterpolator(x_points, x_values, num_threads=num_threads, triangles=triangles)(
            int_points, fill_value=0.0
        )
        scipy_values = griddata(x_points, x_values, int_points, method='linear', fill_value=0.0)

        delta_ds = np.abs(delaunay_values - scipy_values)
        delta_si = np.abs(scipy_values - imops_values)

        assert delta_ds.max() <= 1e-10 and delta_si.max() <= 5, f'Failed with big case, arr_{i}'

    for i in range(2, 7):
        x = load(tests_root / 'test_data' / f'arr_{i}.npy.gz')
        x_values = load(tests_root / 'test_data' / f'val_{i}.npy.gz')
        x_points = np.transpose(x.nonzero())
        int_points = np.transpose((~x).nonzero())
        imops_values = Linear2DInterpolator(x_points, x_values, num_threads=num_threads)(int_points, fill_value=0.0)
        triangles = scipy.spatial.Delaunay(x_points).simplices
        delaunay_values = Linear2DInterpolator(x_points, x_values, num_threads=num_threads, triangles=triangles)(
            int_points, fill_value=0.0
        )
        scipy_values = griddata(x_points, x_values, int_points, method='linear', fill_value=0.0)

        delta_ds = np.abs(delaunay_values - scipy_values)
        delta_di = np.abs(delaunay_values - imops_values)
        delta_si = np.abs(scipy_values - imops_values)

        assert (
            delta_di.max() <= 1.5 and delta_ds.max() <= 1e-10 and delta_si.max() <= 1.5
        ), f'Failed with small case: arr_{i}'


def test_no_values(example):
    x_points, int_points = example

    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points)(int_points)


def test_no_changes_in_values(example):
    x_points, int_points = example
    first_values = np.ones((x_points.shape[0],), dtype=float)
    second_values = 2.0 * np.ones((x_points.shape[0],), dtype=float)
    interpolator = Linear2DInterpolator(x_points, first_values)
    interpolator(int_points, second_values)
    assert np.all(interpolator.values == first_values), 'Failed with changes in self.values after __call__'


def test_bad_values_dtype(example):
    x_points, int_points = example

    with pytest.raises(TypeError):
        Linear2DInterpolator(x_points, values=[1, 2, 3])(int_points)
    with pytest.raises(TypeError):
        Linear2DInterpolator(x_points)(int_points, values=[1, 2, 3])
    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points)(int_points, values=np.ones((3, 3)))


def test_bad_values_lenght(example):
    x_points, int_points = example

    values = np.ones(len(x_points) + 10)
    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points, values=values)(int_points)
    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points)(int_points, values=values)

    values = values[:-20]
    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points, values=values)(int_points)
    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points)(int_points, values=values)


def test_bad_triangles_dtype(example):
    x_points, _ = example

    with pytest.raises(TypeError):
        Linear2DInterpolator(x_points, triangles=[1, 2, 3])
    with pytest.raises(ValueError):
        Linear2DInterpolator(x_points, triangles=np.ones((3, 2)))


def test_bad_points_dtype(example):
    x_points, _ = example

    values = np.ones((x_points.shape[0],))
    with pytest.raises(TypeError):
        Linear2DInterpolator(points=[1, 2, 3])
    with pytest.raises(TypeError):
        Linear2DInterpolator(points=x_points, values=values)(points=[1, 2, 3])
    with pytest.raises(ValueError):
        Linear2DInterpolator(points=np.ones((3, 3)))
    with pytest.raises(ValueError):
        Linear2DInterpolator(points=x_points, values=values)(points=np.ones((3, 3)))
