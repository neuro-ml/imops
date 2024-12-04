# Imops

Efficient parallelizable algorithms for multidimensional arrays to speed up your data pipelines

## Install

```shell
pip install imops  # default install with Cython backend
pip install imops[numba]  # additionally install Numba backend
```

## Functions


::: imops.crop.crop_to_shape

::: imops.crop.crop_to_box

::: imops.pad.pad

::: imops.pad.pad_to_shape

::: imops.pad.pad_to_divisible

::: imops.pad.restore_crop

::: imops.zoom.zoom

::: imops.zoom.zoom_to_shape

::: imops.interp1d.interp1d
    options:
      members:
        - __call__

::: imops.interp2d.Linear2DInterpolator
    options:
      members:
        - __call__

::: imops.morphology.binary_dilation

::: imops.morphology.binary_erosion

::: imops.morphology.binary_opening

::: imops.morphology.binary_closing

::: imops.morphology.distance_transform_edt

::: imops.morphology.convex_hull_image

::: imops.measure.label

::: imops.measure.center_of_mass

::: imops.numeric.pointwise_add

::: imops.numeric.fill_

::: imops.numeric.full

::: imops.numeric.copy

::: imops.radon.radon

::: imops.radon.inverse_radon

::: imops.utils.isin
