import pytest

from imops.utils import build_slices, check_len


def test_check_len():
    assert check_len([]) is None
    assert check_len([0]) is None
    assert check_len([0], [1], [2]) is None

    with pytest.raises(ValueError):
        check_len([], [0])

    with pytest.raises(ValueError):
        check_len([], [0], [1, 2])


def test_build_slices():
    assert build_slices([10, 11]) == (slice(10), slice(11))
    assert build_slices([1, 2, 3], [4, 5, 6]) == (slice(1, 4), slice(2, 5), slice(3, 6))
    assert build_slices([1, 2, 3], [4, 5, 6], [7, 8, 9]) == (slice(1, 4, 7), slice(2, 5, 8), slice(3, 6, 9))

    with pytest.raises(ValueError):
        build_slices([0, 1], [2])
    with pytest.raises(ValueError):
        build_slices([0], [None], [2, 3, 4])
