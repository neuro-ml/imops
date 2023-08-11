import numpy as np
import pytest

from imops.box import add_margin, box_to_shape, make_box, mask_to_box, returns_box, shape_to_box


assert_eq = np.testing.assert_equal


def test_make_box():
    box = make_box(((0, 0), (12, 123)))

    with pytest.raises(ValueError) as e:
        box[0, 0] = 1

    e_text = str(e.value)

    assert 'destination is read-only' in e_text, e_text


def test_return_box():
    box = returns_box(lambda: ((0, 0), (12, 123)))()

    with pytest.raises(ValueError) as e:
        box[0, 0] = 1

    e_text = str(e.value)

    assert 'destination is read-only' in e_text, e_text


def test_mask_to_box():
    mask = np.zeros((10, 10))

    with pytest.raises(ValueError):
        mask_to_box(mask)

    mask[2, 3] = mask[4, 5] = True
    assert_eq(mask_to_box(mask), ((2, 3), (5, 6)))


def test_add_margin():
    box = make_box(((0, 0), (4, 5)))

    assert_eq(box, add_margin(box, [0, 0]))
    assert_eq(make_box(((-1, -1), (5, 6))), add_margin(box, [1, 1]))
    assert_eq(make_box(((-2, -10), (6, 15))), add_margin(box, [2, 10]))


def test_shape_box():
    box = make_box(((1, 2), (3, 4)))

    assert_eq(box_to_shape(box), (2, 2))
    assert_eq(make_box(((0, 0), (2, 2))), shape_to_box(box_to_shape(box)))
