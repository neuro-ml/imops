[![codecov](https://codecov.io/gh/neuro-ml/imops/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/imops)
[![pypi](https://img.shields.io/pypi/v/imops?logo=pypi&label=PyPi)](https://pypi.org/project/imops/)
![License](https://img.shields.io/github/license/neuro-ml/imops)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/imops)](https://pypi.org/project/imops/)

# Imops

Efficient parallelizable algorithms for multidimensional arrays to speed up your data pipelines

# Install

```shell
pip install imops
```

# Features

## Fast Radon transform

```python
from imops import radon, inverse_radon
```

## Fast linear/bilinear/trilinear zoom

```python
from imops import zoom
from imops import zoom_to_shape

# fast zoom with optional fallback to scipy's implementation
y = zoom(x, 2, axis=[0, 1])
# a handy function to zoom the array to a given shape 
# without the need to compute the scale factor
z = zoom_to_shape(x, (4, 120, 67))
```
Works faster only for `ndim<=3, dtype=float32 or float64, output=None, order=1, mode='constant', grid_mode=False`
## Fast 1d linear interpolation

```python
from imops import interp1d  # same as `scipy.interpolate.interp1d`
```
Works faster only for `ndim<=3, dtype=float32 or float64, order=1 or 'linear'`
## Padding

```python
from imops import pad, pad_to_shape

y = pad(x, 10, axis=[0, 1])
# `ratio` controls how much padding is applied to left side:
# 0 - pad from right
# 1 - pad from left
# 0.5 - distribute the padding equally
z = pad_to_shape(x, (4, 120, 67), ratio=0.25)
```

## Cropping

```python
from imops import crop_to_shape

# `ratio` controls the position of the crop
# 0 - crop from right
# 1 - crop from left
# 0.5 - crop from the middle
z = crop_to_shape(x, (4, 120, 67), ratio=0.25)
```

# Acknowledgements

Some parts of our code for radon/inverse radon transform as well as the code for linear interpolation are inspired by
the implementations from [scikit-image](https://github.com/scikit-image/scikit-image)
and [scipy](https://github.com/scipy/scipy).
