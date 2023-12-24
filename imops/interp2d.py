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

    def __init__(self, points: np.ndarray, values: np.ndarray, n_jobs: int = 1,  triangles: Optional[np.ndarray] = None, **kwargs):
        super().__init__(points, n_jobs, triangles)
        self.kdtree = KDTree(data=points, **kwargs)
        self.values = values
        self.n_jobs = n_jobs

    def __call__(
        self, points: np.ndarray, values: Optional[np.ndarray] = None, fill_value: float = 0.0
    ) -> np.ndarray[np.float32]:
        self.values = self.values if values is None else values
        _, neighbors = self.kdtree.query(points, 1, workers=self.n_jobs)
        int_values = super().__call__(points, self.values, neighbors, fill_value)
        return int_values
