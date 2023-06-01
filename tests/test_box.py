import numpy as np
import pytest

from imops.box import (
    add_margin,
    broadcast_box,
    get_box_padding,
    get_centered_box,
    get_containing_box,
    limit_box,
    make_box_,
    mask2bounding_box,
    returns_box,
)


assert_eq = np.testing.assert_equal


def test_make_box():
    box = make_box_(((0, 0), (12, 123)))

    with pytest.raises(ValueError) as e:
        box[0, 0] = 1

    e_text = str(e.value)

    assert 'destination is read-only' in e_text, e_text


def test_broadcast_box():
    spatial_box = make_box_(((2, 3), (5, 5)))
    shape = np.array([3, 10, 10, 10])
    spatial_dims = (-3, -2)

    assert_eq(broadcast_box(spatial_box, shape, spatial_dims), ((0, 2, 3, 0), (3, 5, 5, 10)))


def test_get_containing_box():
    shape = (3, 4, 10, 12)
    assert_eq(get_containing_box(shape), [[0, 0, 0, 0], shape])


def test_return_box():
    box = returns_box(lambda: ((0, 0), (12, 123)))()

    with pytest.raises(ValueError) as e:
        box[0, 0] = 1

    e_text = str(e.value)

    assert 'destination is read-only' in e_text, e_text


def test_limit_full_box():
    limit = (1, 3, 4, 5)
    box = (tuple([0] * len(limit)), tuple(limit))
    assert_eq(limit_box(box, limit), box)


def test_limit_box():
    limit = (10, 10, 10)
    box = ((0, -1, -10), (10, 100, 17))
    assert_eq(limit_box(box, limit), ((0, 0, 0), (10, 10, 10)))


def test_get_box_padding():
    limit = np.array((10, 10, 10))
    box = np.array(((0, -1, -10), (10, 100, 17)))
    assert_eq(get_box_padding(box, limit).T, ((0, 1, 10), (0, 90, 7)))


def test_add_margin():
    box = np.array(((0, -1), (10, 100)))
    margin = 10

    assert_eq(add_margin(box, margin), ((-10, -11), (20, 110)))

    margin = [1, 10]

    assert_eq(add_margin(box, margin), ((-1, -11), (11, 110)))


def test_get_centered_box():
    box_size = np.array((2, 3))
    center = np.array([5, 6])
    assert_eq(get_centered_box(center=center, box_size=box_size), ((4, 5), (6, 8)))

    limit = np.array((15, 16))
    center = limit // 2
    start, stop = get_centered_box(center, limit)

    assert_eq(start, 0)
    assert_eq(stop, limit)


def test_mask2bounding_box():
    mask = np.zeros((10, 10))
    mask[2, 3] = mask[4, 5] = True
    assert_eq(mask2bounding_box(mask), ((2, 3), (5, 6)))
