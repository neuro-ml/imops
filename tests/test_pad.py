from functools import partial

import numpy as np
import pytest

from imops.crop import crop_to_box
from imops.pad import pad, pad_to_divisible, pad_to_shape, restore_crop
from imops.testing import seeded_by


SEED = 1337
assert_eq = np.testing.assert_array_equal


# TODO: Add more comprehensive tests
@seeded_by(SEED)
def test_broadcasting():
    x = np.random.randint(0, 100, (3, 20, 23))
    main = pad(x, [[3, 3], [3, 3], [3, 3]])

    assert_eq(x, main[3:-3, 3:-3, 3:-3])
    assert_eq(main, pad(x, [3, 3, 3]))
    assert_eq(main, pad(x, 3, axis=[0, 1, 2]))
    assert_eq(main, pad(x, [3], axis=[0, 1, 2]))
    assert_eq(main, pad(x, [[3]], axis=[0, 1, 2]))
    assert_eq(main, pad(x, [[3, 3]], axis=[0, 1, 2]))
    assert_eq(main, pad(x, [[3], [3], [3]], axis=[0, 1, 2]))

    assert_eq(
        pad(x, 3, axis=[0, 1]),
        pad(x, [[3, 3], [3, 3], [0, 0]]),
    )
    assert_eq(
        pad(x, [2, 4, 3]),
        pad(x, [[2, 2], [4, 4], [3, 3]]),
    )
    p = pad(x, [[1, 2], [3, 4], [5, 6]])
    assert_eq(x, p[1:-2, 3:-4, 5:-6])

    p = pad(x, [[1, 2], [3, 4]], axis=[0, 2])
    assert_eq(x, p[1:-2, :, 3:-4])

    p = pad(x, [[1, 2], [3, 4]], axis=[2, 0])
    assert_eq(x, p[3:-4:, :, 1:-2])

    with pytest.raises(ValueError):
        pad(x, [1, 2], axis=-1)


@seeded_by(SEED)
def test_padding_values():
    x = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )

    p = pad(x, [1, 1], padding_values=1)
    assert_eq(
        p,
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
    )

    x = np.random.randint(0, 100, (3, 20, 23))
    assert_eq(
        pad(x, [1, 1], padding_values=x.min(), axis=(1, 2)),
        pad(x, [1, 1], padding_values=np.min, axis=(1, 2)),
    )
    assert_eq(
        pad(x, [1, 1], padding_values=x.min(axis=(1, 2), keepdims=True), axis=(1, 2)),
        pad(x, [1, 1], padding_values=partial(np.min, axis=(1, 2), keepdims=True), axis=(1, 2)),
    )


def test_pad():
    x = np.arange(12).reshape((3, 2, 2))
    padding = np.array(((0, 0), (1, 2), (2, 1)))
    padding_values = np.min(x, axis=(1, 2), keepdims=True)

    y = pad(x, padding, padding_values=padding_values)
    assert_eq(
        y,
        np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 2, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [4, 4, 4, 4, 4],
                    [4, 4, 4, 5, 4],
                    [4, 4, 6, 7, 4],
                    [4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4],
                ],
                [
                    [8, 8, 8, 8, 8],
                    [8, 8, 8, 9, 8],
                    [8, 8, 10, 11, 8],
                    [8, 8, 8, 8, 8],
                    [8, 8, 8, 8, 8],
                ],
            ]
        ),
    )


@seeded_by(SEED)
def test_pad_to_shape():
    x = np.random.rand(3, 10, 10)
    shape = 3, 15, 16
    assert pad_to_shape(x, shape).shape == shape
    with pytest.raises(ValueError):
        pad_to_shape(x, (3, 4, 10))


@seeded_by(SEED)
def test_restore_crop():
    x = np.random.rand(3, 10, 10)
    box = [1, 2, 3], [3, 8, 9]

    assert (restore_crop(crop_to_box(x, box), box, x.shape)).shape == x.shape


def test_restore_crop_invalid_box():
    x = np.ones((2, 3, 4))

    with pytest.raises(ValueError):
        restore_crop(x, np.array([[0, 0, 0], [1, 1, 1]]), [4, 4, 4])

    with pytest.raises(ValueError):
        restore_crop(x, np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]), [4.5, 4.5, 4.5])


def test_pad_to_divisible():
    x = np.zeros((4, 8, 12))

    y = pad_to_divisible(x, 4)
    assert_eq(y, x)

    x = np.zeros((3, 5))

    y = pad_to_divisible(x, 1)
    assert_eq(y, x)

    y = pad_to_divisible(x, 4)
    assert_eq(y, np.zeros((4, 8)))

    y = pad_to_divisible(x, 4, remainder=2)
    assert_eq(y, np.zeros((6, 6)))


def test_negative_padding():
    x = np.zeros((3, 4, 5))

    with pytest.raises(ValueError):
        pad(x, padding=-1)
