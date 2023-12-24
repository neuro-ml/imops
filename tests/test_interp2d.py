import numpy as np
import scipy
from scipy.interpolate import griddata

from imops.interp2d import Linear2DInterpolator
from deli import load_numpy

np.random.seed(1337)
n_jobs = 1

def test_test_data():
    for i in range(2):
        x = load_numpy(f'test_data/arr_{i}.npy')
        int_points = np.transpose((np.isnan(x)).nonzero())
        x_points: np.ndarray = np.transpose((~np.isnan(x)).nonzero())
        x_values = x[~np.isnan(x)]
        imops_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs)(int_points, fill_value=0.0)
        triangles = scipy.spatial.Delaunay(x_points).simplices
        delaunay_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs, triangles=triangles)(int_points, fill_value=0.0)
        scipy_values = griddata(x_points, x_values, int_points, method='linear', fill_value=0.0)

        delta_ds = np.abs(delaunay_values - scipy_values)
        # delta_di = np.abs(delaunay_values - imops_values)
        delta_si = np.abs(scipy_values - imops_values)
        
        assert delta_ds.max() <= 1e-10 and delta_si.max() <= 5, 'Failed with big cases, arr_0 or arr_1'
    
    for i in range(2, 7):
        x = load_numpy(f'test_data/arr_{i}.npy')
        x_values = load_numpy(f'test_data/val_{i}.npy')
        x_points = np.transpose(x.nonzero())
        int_points = np.transpose((~x).nonzero())
        imops_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs)(int_points, fill_value=0.0)
        triangles = scipy.spatial.Delaunay(x_points).simplices
        delaunay_values = Linear2DInterpolator(x_points, x_values, n_jobs=n_jobs, triangles=triangles)(int_points, fill_value=0.0)
        scipy_values = griddata(x_points, x_values, int_points, method='linear', fill_value=0.0)

        delta_ds = np.abs(delaunay_values - scipy_values)
        delta_di = np.abs(delaunay_values - imops_values)
        delta_si = np.abs(scipy_values - imops_values)

        assert delta_di.max() <= 1.5 and delta_ds.max() <= 1e-10 and delta_si.max() <= 1.5, 'Failed with small cases, arr_2 ...'
