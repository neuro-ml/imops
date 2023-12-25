import numpy as np
from scipy.spatial import KDTree
from platform import python_version

# TODO: can we make it submodule like `from imops.cpp import Linear2DInterpolatorCpp`?
from cpp_modules import Linear2DInterpolatorCpp

from .utils import normalize_num_threads
from .backend import Cython


class Linear2DInterpolator(Linear2DInterpolatorCpp):
    """
    2D Delaunay triangulation and parallel linear interpolation

    Parameters
    ----------
    points: np.ndarray
        2-D array of data point coordinates
    values: np.ndarray
        1-D array of fp32/fp64 values
    num_threads: int
        the number of threads to use for computation. Default = the cpu count. If negative value passed
        cpu count + num_threads + 1 threads will be used
    triangels: np.ndarray
        optional precomputed triangulation in the form of array or arrays of points indices

    Methods
    -------
    __call__

    Examples
    --------
    >>> n, m = 1024, 2
    >>> points = np.random.randint(low=0, high=1024, size=(n, m))
    >>> points = np.unique(points, axis=0)
    >>> x_points = points[: n // 2]
    >>> values = np.random.uniform(low=0.0, high=1.0, size=(len(x_points),))
    >>> interp_points = points[n // 2:]
    >>> num_threads = -1 // will be equal to num of CPU cores
    >>> interpolator = Linear2DInterpolator(x_points, values, num_threads)
    >>> // Also you can pass values to __call__ and rewrite the ones that were passed to __init__
    >>> interp_values = interpolator(interp_points, values + 1.0, fill_value=0.0)
    """

    def __init__(
        self, points: np.ndarray, values: np.ndarray, num_threads: int = 1, triangles: np.ndarray = None, **kwargs
    ):
        if triangles is not None:
            if isinstance(triangles, np.ndarray):
                if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] * 3 != triangles.size:
                    raise ValueError('Passed \"triangles\" argument has an incorrect shape')
            else:
                raise ValueError(f'Wrong type of \"triangles\" argument, expected np.ndarray. Got {type(triangles)}')

        super().__init__(points, num_threads, triangles)
        self.kdtree = KDTree(data=points, **kwargs)
        self.values = values
        # FIXME: add backend dispatch
        self.num_threads = normalize_num_threads(num_threads, Cython(), warn_stacklevel=3)

    def __call__(self, points: np.ndarray, values: np.ndarray = None, fill_value: float = 0.0) -> np.ndarray:
        """
        Evaluate the interpolant

        Parameters
        ----------
        points: np.ndarray
            2-D array of data point coordinates to interpolate at
        values: np.ndarray
            1-D array of fp32/fp64 values to use at initial points. If passed, existing values will be rewritten
        fill_value: float
            value to fill past edges

        Returns
        -------
        new_values: np.ndarray
            interpolated values at given points
        """
        if values is not None:
            if isinstance(values, np.ndarray):
                if values.ndim > 1:
                    raise ValueError(f'Wrong shape of \"values\" argument, expected ndim=1. Got shape {values.shape}')
            else:
                raise ValueError(f'Wrong type of \"values\" argument, expected np.ndarray. Got {type(values)}')

        self.values = self.values if values is None else values

        if self.values is None:
            raise ValueError('\"values\" argument was never passed neither in __init__ or __call__ methods')

        _, neighbors = self.kdtree.query(
            points, 1, **{'workers': self.num_threads} if python_version()[:3] != '3.6' else {}
        )

        if not isinstance(points, np.ndarray):
            raise ValueError(f'Wrong type of \"points\" argument, expected np.ndarray. Got {type(points)}')

        int_values = super().__call__(points, self.values, neighbors, fill_value)
        return int_values
