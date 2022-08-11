# Imops
Efficient parallelizable algorithms to speed up your data pipelines
# Install
```
git clone <repo>
cd <repo name>
pip install -e .
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
