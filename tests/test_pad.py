from functools import partial

import numpy as np
import pytest

from imops import pad


assert_eq = np.testing.assert_array_equal


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
    np.testing.assert_array_equal(
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
