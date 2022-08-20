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

# Acknowledgements

Some parts of our code for radon/inverse radon transform as well as the code for linear interpolation are inspired by
the implementations from [scikit-image](https://github.com/scikit-image/scikit-image)
and [scipy](https://github.com/scipy/scipy).
