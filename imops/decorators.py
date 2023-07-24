from typing import Callable, Union

import numpy as np

from .box import Box
from .crop import crop_to_box
from .pad import restore_crop
from .utils import AxesParams


def crop_apply_pad(func: Callable, padding_values: Union[AxesParams, Callable] = 0) -> Callable:
    """
    Modifies passed function so it could crop input image to the specified box, operate on cropped image and pad to
    the initial shape. This may improve performance depending on the size of the cropped image, e.g. for morphology ops
    applied to masks of small localized objects.

    Parameters
    ----------
    func: Callable
        function to decorate
    padding_values: Union[AxesParams, Callable]
        values to pad with after applying `func`, must be broadcastable to the resulting array.
        If Callable (e.g. `numpy.min`) - `padding_values(x)` will be used

    Returns
    -------
    decorated: Callable
        decorated function

    Examples
    --------
    >>> from imops.box import mask_to_box, dilate_box
    >>> from imops.morphology import binary_erosion
    >>> # input should be cropped to the dilated box to erode it properly
    >>> dilated = crop_apply_pad(binary_erosion)(x, box=dilate_box(mask_to_box(x), 1, x.shape))
    """

    def decorated(
        x: np.ndarray,
        *func_args,
        box: Box = None,
        **func_kwargs,
    ) -> np.ndarray:
        if box is None:
            return func(x, *func_args, **func_kwargs)

        shape = x.shape

        return restore_crop(
            func(crop_to_box(x, box, padding_values=None), *func_args, **func_kwargs),
            box,
            shape,
            padding_values=padding_values(x) if callable(padding_values) else padding_values,
        )

    return decorated
