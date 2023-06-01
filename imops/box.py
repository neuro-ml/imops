import itertools
from functools import wraps
from typing import Callable, Sequence, Tuple

import numpy as np


Box = np.ndarray


def check_len(*args) -> None:
    lengths = list(map(len, args))
    if any(length != lengths[0] for length in lengths):
        raise ValueError(f'Arguments of equal length are required: {", ".join(map(str, lengths))}')


def build_slices(start: Sequence[int], stop: Sequence[int] = None) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices built from `start` and `stop`.

    Examples
    --------
    >>> build_slices([1, 2, 3], [4, 5, 6])
    (slice(1, 4), slice(2, 5), slice(3, 6))
    >>> build_slices([10, 11])
    (slice(10), slice(11))
    """
    if stop is not None:
        check_len(start, stop)
        return tuple(map(slice, start, stop))

    return tuple(map(slice, start))


def compute_shape_from_spatial(complete_shape, spatial_shape, spatial_dims):
    check_len(spatial_shape, spatial_dims)
    shape = np.array(complete_shape)
    shape[list(spatial_dims)] = spatial_shape

    return tuple(shape)


def make_box_(iterable) -> Box:
    """
    Returns a box, generated inplace from the `iterable`. If `iterable` was a numpy array, will make it
    immutable and return.
    """
    box = np.asarray(iterable)
    box.setflags(write=False)

    assert box.ndim == 2 and len(box) == 2, box.shape
    assert np.all(box[0] <= box[1]), box

    return box


def get_volume(box: Box):
    return np.prod(box[1] - box[0], axis=0)


def returns_box(func: Callable) -> Callable:
    """Returns function, decorated so that it returns a box."""

    @wraps(func)
    def func_returning_box(*args, **kwargs):
        return make_box_(func(*args, **kwargs))

    func_returning_box.__annotations__['return'] = Box

    return func_returning_box


@returns_box
def get_containing_box(shape: tuple) -> Box:
    """Returns box that contains complete array of shape `shape`."""
    return [0] * len(shape), shape


@returns_box
def broadcast_box(box: Box, shape: tuple, dims: tuple) -> Box:
    """
    Returns box, such that it contains `box` across `dims` and whole array
    with shape `shape` across other dimensions.
    """
    return (compute_shape_from_spatial([0] * len(shape), box[0], dims), compute_shape_from_spatial(shape, box[1], dims))


@returns_box
def limit_box(box, limit) -> Box:
    """
    Returns a box, maximum subset of the input `box` so that start would be non-negative and
    stop would be limited by the `limit`.
    """
    check_len(*box, limit)
    return np.maximum(box[0], 0), np.minimum(box[1], limit)


def get_box_padding(box: Box, limit):
    """
    Returns padding that is necessary to get `box` from array of shape `limit`.
    Returns padding in numpy form, so it can be given to `numpy.pad`.
    """
    check_len(*box, limit)
    return np.maximum([-box[0], box[1] - limit], 0).T


@returns_box
def get_union_box(*boxes) -> Box:
    start = np.min([box[0] for box in boxes], axis=0)
    stop = np.max([box[1] for box in boxes], axis=0)
    return start, stop


@returns_box
def add_margin(box: Box, margin) -> Box:
    """
    Returns a box with size increased by the `margin` (need to be broadcastable to the box)
    compared to the input `box`.
    """
    margin = np.broadcast_to(margin, box.shape)
    return box[0] - margin[0], box[1] + margin[1]


@returns_box
def get_centered_box(center: np.ndarray, box_size: np.ndarray) -> Box:
    """
    Get box of size `box_size`, centered in the `center`.
    If `box_size` is odd, `center` will be closer to the right.
    """
    start = center - box_size // 2
    stop = center + box_size // 2 + box_size % 2

    return start, stop


@returns_box
def mask2bounding_box(mask: np.ndarray) -> Box:
    """
    Find the smallest box that contains all true values of the `mask`.
    """
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


def box2slices(box: Box) -> Tuple[slice, ...]:
    return build_slices(*box)
