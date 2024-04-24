import numpy as np
import pytest

from imops.crop import crop_to_box, crop_to_shape
from imops.testing import seeded_by


SEED = 1337
assert_eq = np.testing.assert_array_equal


@seeded_by(SEED)
def test_shape():
    for _ in range(100):
        shape = np.random.randint(10, 50, size=2)
        box_shape = np.random.randint(1, 50, size=2)
        box_center = [np.random.randint(s) for s in shape]
        start = box_center - box_shape // 2

        x = np.empty(shape)
        box = np.stack([start, start + box_shape])

        assert (crop_to_box(x, box, padding_values=0).shape == box_shape).all()


@seeded_by(SEED)
def test_axes():
    x = np.random.randint(0, 100, (3, 20, 23))

    assert_eq(x[:, 1:15, 2:14], crop_to_box(x, np.array([[1, 2], [15, 14]]), axis=[1, 2]))

    assert_eq(
        x[:, 1:, 2:],
        crop_to_box(x, np.array([[1, 2], [40, 33]]), padding_values=0, axis=(1, 2))[:, :19, :21],
    )

    assert_eq(
        x[:, :15, :14],
        crop_to_box(x, np.array([[-10, -5], [15, 14]]), padding_values=0, axis=(1, 2))[:, 10:, 5:],
    )


def test_raises():
    x = np.empty((3, 20, 23))
    with pytest.raises(ValueError):
        crop_to_box(x, np.array([[1], [40]]), axis=(1, 2))

    with pytest.raises(ValueError):
        crop_to_box(x, np.array([[-1], [1]]), axis=(1, 2))


@seeded_by(SEED)
def test_crop_to_shape():
    x = np.random.rand(3, 10, 10)
    shape = (3, 4, 8)
    assert crop_to_shape(x, shape).shape == shape
    with pytest.raises(ValueError):
        crop_to_shape(x, (3, 15, 10))


def test_crop_to_float_shape():
    x = np.random.rand(3, 10, 10)
    float_shape = (1.337, 3.1415, 2.7182)
    with pytest.raises(ValueError):
        crop_to_shape(x, float_shape)


def test_crop_to_float_box():
    x = np.random.rand(3, 10, 10)
    float_box = [[0, 1], [4, 4.5], [3.1, 9]]
    with pytest.raises(ValueError):
        crop_to_box(x, float_box)
