[![codecov](https://codecov.io/gh/neuro-ml/imops/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/imops)
[![pypi](https://img.shields.io/pypi/v/imops?logo=pypi&label=PyPi)](https://pypi.org/project/imops/)
![License](https://img.shields.io/github/license/neuro-ml/imops)

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
#  withtout the need to compute the scale factor
z = zoom_to_shape(x, (4, 120, 67))
```

## Fast 1d linear interpolation

```
from imops import interp1d  # Same as `scipy.interpolate.interp1d`
```

Note: interp1d works only with FP32 / FP64 inputs

## Padding

```python
from imops import pad, pad_to_shape

y = pad(x, 10, axis=[0, 1])
# `ratio` controls how much padding is applied for each side:
#  0 - pad from left
#  1 - pad from right
#  0.5 - distribute the padding equally
z = pad_to_shape(x, (4, 120, 67), ratio=0.25)
```

## Cropping

```python
from imops import crop_to_shape

# `ratio` controls the position of the crop
#  0 - crop from left
#  1 - crop from right
#  0.5 - crop from the middle
z = crop_to_shape(x, (4, 120, 67), ratio=0.25)
```

# Acknowledgements

Some parts of our code for radon/inverse radon transform as well as the code for linear interpolation are inspired by
the implementations from [scikit-image](https://github.com/scikit-image/scikit-image)
and [scipy](https://github.com/scipy/scipy).
