from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from cpp_modules import Linear2DInterpolatorCpp


class Linear2DInterpolator(Linear2DInterpolatorCpp):
    # """
    # #### 2D delaunay triangulation and parallel linear interpolation
    # Example:
    # n, m = 1024, 2
    # points = np.random.randint(low=0, high=1024, size=(n, m))
    # points = np.unique(points, axis=0)
    # x_points = points[: n // 2]
    # values = np.random.uniform(low=0.0, high=1.0, size=(len(x_points),))
    # interp_points = points[n // 2:]
    # n_jobs = -1 // will be equal to num of CPU cores
    # interpolator = Linear2DInterpolator(x_points, values, n_jobs)
    # // Also you can pass values to __call__ and rewrite the ones that were passed to __init__
    # interp_values = interpolator(interp_points, values + 1.0, fill_value=0.0)
    # """

    def __init__(
        self, points: np.ndarray, values: np.ndarray, n_jobs: int = 1, triangles: Optional[np.ndarray] = None, **kwargs
    ):

        if triangles is not None:
            if isinstance(triangles, np.ndarray):
                if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] * 3 != triangles.size:
                    raise ValueError('Passed \"triangles\" argument has an incorrect shape')
            else:
                raise ValueError(
                    f'Wrong type of \"triangles\" argument, expected Optional[np.ndarray]. Got {type(triangles)}'
                )

        if n_jobs <= 0 and n_jobs != -1:
            raise ValueError('Invalid number of workers, has to be -1 or positive integer')

        super().__init__(points, n_jobs, triangles)
        self.kdtree = KDTree(data=points, **kwargs)
        self.values = values
        self.n_jobs = n_jobs

    def __call__(
        self, points: np.ndarray, values: Optional[np.ndarray] = None, fill_value: float = 0.0
    ) -> np.ndarray[np.float32]:

        if values is not None:
            if isinstance(values, np.ndarray):
                if values.ndim > 1:
                    raise ValueError(f'Wrong shape of \"values\" argument, expected ndim=1. Got shape {values.shape}')
            else:
                raise ValueError(
                    f'Wrong type of \"values\" argument, expected Optional[np.ndarray]. Got {type(values)}'
                )

        self.values = self.values if values is None else values

        if self.values is None:
            raise ValueError('\"values\" argument was never passed neither in __init__ or __call__ methods')

        _, neighbors = self.kdtree.query(points, 1, workers=self.n_jobs)

        if not isinstance(points, np.ndarray):
            raise ValueError(f'Wrong type of \"points\" argument, expected np.ndarray. Got {type(points)}')

        int_values = super().__call__(points, self.values, neighbors, fill_value)
        return int_values
