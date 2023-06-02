import numpy as np
import pytest

from imops.box import make_box, mask_to_box, returns_box


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
    mask[2, 3] = mask[4, 5] = True
    assert_eq(mask_to_box(mask), ((2, 3), (5, 6)))
