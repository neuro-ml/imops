import os
from unittest import mock

import numpy as np
import pytest

from imops.backend import Cython
from imops.utils import (
    broadcast_axis,
    broadcast_to_axis,
    build_slices,
    check_len,
    imops_num_threads,
    isin,
    make_immutable,
    make_mutable,
    normalize_num_threads,
    set_num_threads,
)


assert_eq = np.testing.assert_array_equal

MANY_THREADS = 42069


@pytest.fixture(params=[np.int16, np.int32, np.int64])
def dtype(request):
    return request.param


@pytest.fixture(params=[bool, np.float32, np.uint8])
def bad_dtype(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def num_threads(request):
    return request.param


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


@mock.patch.dict(os.environ, {}, clear=True)
def test_set_num_threads():
    PREV_IMOPS_NUM_THREADS = set_num_threads(10)

    assert PREV_IMOPS_NUM_THREADS is None
    assert normalize_num_threads(-1, Cython()) == min(10, len(os.sched_getaffinity(0)))

    PREV_IMOPS_NUM_THREADS = set_num_threads(None)
    assert PREV_IMOPS_NUM_THREADS == 10
    assert normalize_num_threads(-1, Cython()) == len(os.sched_getaffinity(0))


@mock.patch.dict(os.environ, {}, clear=True)
def test_imops_num_threads():
    with imops_num_threads(10):
        assert normalize_num_threads(-1, Cython()) == min(10, len(os.sched_getaffinity(0)))

    assert normalize_num_threads(-1, Cython()) == len(os.sched_getaffinity(0))


@mock.patch.dict(os.environ, {}, clear=True)
def test_many_threads_warning_os():
    with pytest.warns(UserWarning):
        normalize_num_threads(MANY_THREADS, Cython())


@mock.patch.dict(os.environ, {'OMP_NUM_THREADS': '2'}, clear=True)
def test_many_threads_warning_omp():
    with pytest.warns(UserWarning):
        normalize_num_threads(MANY_THREADS, Cython())


@mock.patch.dict(os.environ, {}, clear=True)
def test_many_threads_warning_imops():
    with imops_num_threads(10):
        with pytest.warns(UserWarning):
            normalize_num_threads(MANY_THREADS, Cython())


def test_broadcast_to_axis():
    arrays = np.ones((1, 2)), np.ones((3, 4, 5)), np.ones(1), 1
    axis = [0, 0, 0]

    for x, out in zip((np.ones((3, 2)), np.ones((3, 4, 5)), np.ones(3), np.ones(3)), broadcast_to_axis(axis, *arrays)):
        assert_eq(x, out)

    with pytest.raises(ValueError):
        broadcast_to_axis(axis)

    with pytest.raises(ValueError):
        broadcast_to_axis(None, *arrays)

    with pytest.raises(ValueError):
        broadcast_to_axis([0, 0], *arrays)


def test_broadcast_axis():
    arrays = np.ones((1, 3)), np.ones((2, 3))

    for out in broadcast_axis([0, 1], 2, *arrays)[1:]:
        assert_eq(out, np.ones((2, 3)))

    arrays = np.ones((3, 1)), np.ones((2, 3))

    with pytest.raises(ValueError):
        broadcast_axis([0, 1], 2, *arrays)


def test_isin(num_threads, dtype):
    for _ in range(16):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
        elements = np.random.randint(0, 5, size=shape, dtype=dtype)
        test_elements = np.random.randint(0, 5, size=np.random.randint(1, 10))

        if np.random.binomial(1, 0.5):
            make_immutable(elements)
        if np.random.binomial(1, 0.5):
            make_immutable(test_elements)

        assert (np.isin(elements, test_elements) == isin(elements, test_elements, num_threads)).all()


def test_bad_dtype(num_threads, bad_dtype):
    shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
    elements = np.random.randint(0, 5, size=shape).astype(bad_dtype)
    test_elements = np.random.randint(0, 5, size=np.random.randint(1, 10))

    with pytest.raises(ValueError):
        isin(elements, test_elements, num_threads)


def test_make_immutable():
    x = np.ones(3)
    make_immutable(x)
    with pytest.raises(ValueError):
        x[0] = 1337


def test_make_mutable():
    x = np.ones(3)
    make_immutable(x)
    make_mutable(x)
    x[0] = 1337
