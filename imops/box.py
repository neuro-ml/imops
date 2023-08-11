import itertools
from copy import copy
from functools import wraps
from typing import Callable, Tuple

import numpy as np


# Immutable numpy array
Box = np.ndarray


def make_box(iterable) -> Box:
    """Returns a box, generated from copy of the `iterable`."""
    box = np.asarray(copy(iterable))
    box.setflags(write=False)

    assert box.ndim == 2 and len(box) == 2, box.shape
    assert np.all(box[0] <= box[1]), box

    return box


def returns_box(func: Callable) -> Callable:
    """Returns function, decorated so that it returns a box."""

    @wraps(func)
    def func_returning_box(*args, **kwargs):
        return make_box(func(*args, **kwargs))

    func_returning_box.__annotations__['return'] = Box

    return func_returning_box


@returns_box
def mask_to_box(mask: np.ndarray) -> Box:
    """Find the smallest box that contains all true values of the `mask`."""
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)

    return start, stop


@returns_box
def shape_to_box(shape: Tuple) -> Box:
    return make_box([(0,) * len(shape), shape])  # fmt: skip


def box_to_shape(box: Box) -> Tuple:
    return tuple(box[1] - box[0])


@returns_box
def add_margin(box: Box, margin) -> Box:
    """
    Returns a box with size increased by the ``margin`` (need to be broadcastable to the box)
    compared to the input ``box``.
    """
    margin = np.broadcast_to(margin, box.shape)
    return box[0] - margin[0], box[1] + margin[1]
