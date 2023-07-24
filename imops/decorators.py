from typing import Callable, Union

import numpy as np

from .box import Box
from .crop import crop_to_box
from .pad import restore_crop
from .utils import AxesParams


def crop_apply_pad(func, padding_values: Union[AxesParams, Callable] = 0) -> Callable:
    def inner(
        x: np.ndarray,
        *func_args,
        box: Box = None,
        **func_kwargs,
    ) -> np.ndarray:
        if box is None:
            return func(x, *func_args, **func_kwargs)

        shape = x.shape

        return restore_crop(
            func(crop_to_box(x, box, padding_values=padding_values), *func_args, **func_kwargs),
            box,
            shape,
            padding_values,
        )

    return inner
