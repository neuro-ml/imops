# Imops

Efficient parallelizable algorithms to speed up your data pipelines

# Install

```shell
pip install imops
```

# Features

## Fast Radon transform

```
from imops import radon, inverse_radon
```

## Fast linear/bilinear/trilinear zoom

```
from imops import zoom  # Same as `dpipe.im.shape_ops.zoom`
from imops import zoom_to_shape  # Same as `dpipe.im.shape_ops.zoom_to_shape`
from imops import _zoom  # Same as `scipy.ndimage.zoom`
```

Note: all zooms work only with FP32 / FP64 inputs

## Fast 1d linear interpolation

```
from imops import interp1d  # Same as `scipy.interpolate.interp1d`
```

Note: interp1d works only with FP32 / FP64 inputs

# Acknowledgements

Some parts of our code for radon/inverse radon transform as well as the code for linear interpolation are inspired by
the implementations from [scikit-image](https://github.com/scikit-image/scikit-image)
and [scipy](https://github.com/scipy/scipy).
