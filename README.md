[![codecov](https://codecov.io/gh/neuro-ml/imops/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/imops)
[![pypi](https://img.shields.io/pypi/v/imops?logo=pypi&label=PyPi)](https://pypi.org/project/imops/)
![License](https://img.shields.io/github/license/neuro-ml/imops)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/imops)](https://pypi.org/project/imops/)

# Imops

Efficient parallelizable algorithms for multidimensional arrays to speed up your data pipelines.
- [Documentation](https://neuro-ml.github.io/imops/)
- [Benchmarks](https://neuro-ml.github.io/imops/benchmarks/)

# Install

```shell
pip install imops  # default install with Cython backend
```

# How fast is it?

Time comparisons (ms) for Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz using 8 threads. All inputs are C-contiguous NumPy arrays. For morphology functions `bool` dtype is used and `float64` for all others.
| function / backend   |  Scipy()  |  Cython(fast=False)  |  Cython(fast=True)  |
|:----------------------:|:-----------:|:----------------------:|:---------------------:|
| `zoom(..., order=0)` |   2072    |         1114         |         **867**         |
| `zoom(..., order=1)` |   6527    |         596          |         **575**         |
| `interp1d`           |    780    |         149          |         **146**         |
| `radon`              |   59711   |         5982         |        **4837**         |
| `inverse_radon`      |   52928   |         8254         |        **6535**         |
| `binary_dilation`    |   2207    |         310          |         **298**         |
| `binary_erosion`     |   2296    |         326          |         **304**         |
| `binary_closing`     |   4158    |         544          |         **469**         |
| `binary_opening`     |   4410    |         567          |         **522**         |
| `center_of_mass`     |   2237    |          **64**          |         **64**          |

We use [`airspeed velocity`](https://asv.readthedocs.io/en/stable/) to benchmark our code. For detailed results visit [benchmark page](https://neuro-ml.github.io/imops/benchmarks/).

# Features

### Fast Radon transform

```python
from imops import radon, inverse_radon
```

### Fast 0/1-order zoom

```python
from imops import zoom, zoom_to_shape

# fast zoom with optional fallback to scipy's implementation
y = zoom(x, 2, axis=[0, 1])
# a handy function to zoom the array to a given shape 
# without the need to compute the scale factor
z = zoom_to_shape(x, (4, 120, 67))
```
Works faster only for `ndim<=4, dtype=float32 or float64 (and bool-int16-32-64-uint8-16-32 if order == 0), output=None, order=0 or 1, mode='constant', grid_mode=False`
### Fast 1d linear interpolation

```python
from imops import interp1d  # same as `scipy.interpolate.interp1d`
```
Works faster only for `ndim<=3, dtype=float32 or float64, order=1`

### Fast 2d linear interpolation
```python
import numpy as np
from imops.interp2d import Linear2DInterpolator
n, m = 1024, 2
points = np.random.randint(low=0, high=1024, size=(n, m))
points = np.unique(points, axis=0)
x_points = points[: n // 2]
values = np.random.uniform(low=0.0, high=1.0, size=(len(x_points),))
interp_points = points[n // 2:]
num_threads = -1 # will be equal to num of CPU cores
# You can optionally pass your own triangulation as an np.array of shape [num_triangles, 3], element at (i, j) position is an index of a point from x_points
interpolator = Linear2DInterpolator(x_points, values, num_threads=num_threads, triangles=None)
# Also you can pass values to __call__ and rewrite the ones that were passed to __init__
interp_values = interpolator(interp_points, values + 1.0, fill_value=0.0)
```

### Fast binary morphology

```python
from imops import binary_dilation, binary_erosion, binary_opening, binary_closing
```
These functions mimic `scikit-image` counterparts
### Padding

```python
from imops import pad, pad_to_shape

y = pad(x, 10, axis=[0, 1])
# `ratio` controls how much padding is applied to left side:
# 0 - pad from right
# 1 - pad from left
# 0.5 - distribute the padding equally
z = pad_to_shape(x, (4, 120, 67), ratio=0.25)
```

### Cropping

```python
from imops import crop_to_shape

# `ratio` controls the position of the crop
# 0 - crop from right
# 1 - crop from left
# 0.5 - crop from the middle
z = crop_to_shape(x, (4, 120, 67), ratio=0.25)
```

### Labeling

```python
from imops import label

# same as `skimage.measure.label`
labeled, num_components = label(x, background=1, return_num=True)
```

# Backends
For all heavy image routines except `label` you can specify which backend to use. Backend can be specified by a string or by an instance of `Backend` class. The latter allows you to customize some backend options:
```python
from imops import Cython, Scipy, zoom

y = zoom(x, 2, backend='Cython')
y = zoom(x, 2, backend=Cython(fast=False))  # same as previous
y = zoom(x, 2, backend=Cython(fast=True))  # -ffast-math compiled cython backend
y = zoom(x, 2, backend=Scipy())  # use scipy original implementation
```
Also backend can be specified globally or locally:
```python
from imops import imops_backend, set_backend, zoom

set_backend('Scipy')  # sets Scipy as default backend
with imops_backend('Cython'):  # sets Cython backend via context manager
    zoom(x, 2)
```
Available backends:
|         function / backend            | Scipy   | Cython  |
|:-------------------:|:---------:|:---------:|
| `zoom`            | &check; | &check; |
| `interp1d`        | &check; | &check; |
| `radon`           | &cross; | &check; |
| `inverse_radon`   | &cross; | &check; |
| `binary_dilation` | &check; | &check; |
| `binary_erosion`  | &check; | &check; |
| `binary_closing`  | &check; | &check; |
| `binary_opening`  | &check; | &check; |
| `center_of_mass`  | &check; | &check; |

# Acknowledgements

Some parts of our code for radon/inverse radon transform as well as the code for linear interpolation are inspired by
the implementations from [`scikit-image`](https://github.com/scikit-image/scikit-image) and [`scipy`](https://github.com/scipy/scipy).
Also we used [`fastremap`](https://github.com/seung-lab/fastremap), [`edt`](https://github.com/seung-lab/euclidean-distance-transform-3d) and [`cc3d`](https://github.com/seung-lab/connected-components-3d) out of the box.
