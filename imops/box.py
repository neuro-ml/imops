import itertools
from copy import copy
from functools import wraps
from typing import Callable, Sequence, Tuple

import numpy as np


# Immutable numpy array
Box = np.ndarray


def check_len(*args) -> None:
    lengths = list(map(len, args))
    if any(length != lengths[0] for length in lengths):
        raise ValueError(f'Arguments of equal length are required: {", ".join(map(str, lengths))}')


def build_slices(start: Sequence[int], stop: Sequence[int] = None, step: Sequence[int] = None) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices built from `start` and `stop` with `step`.

    Examples
    --------
    >>> build_slices([1, 2, 3], [4, 5, 6])
    (slice(1, 4), slice(2, 5), slice(3, 6))
    >>> build_slices([10, 11])
    (slice(10), slice(11))
    """

    check_len(*filter(lambda x: x is not None, [start, stop, step]))
    args = [
        start,
        stop if stop is not None else [None for _ in start],
        step if step is not None else [None for _ in start],
    ]

    return tuple(map(slice, *args))


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
