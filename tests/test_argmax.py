import numpy as np
import pytest

from imops.argmax import argmax


N_STRESS = 10
np.random.seed(1337)
dim = 13


@pytest.fixture(params=[np.float32, np.float64, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32])
def dtype(request):
    return request.param


@pytest.fixture(params=[1, 2, 7, 54, 128])
def pre_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 7, 54, 128])
def post_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 7, 11, 57, 129])
def argmax_dim(request):
    return request.param


def test_argmax(pre_dim, argmax_dim, post_dim, dtype):
    for _ in range(N_STRESS):
        arr = np.random.randn(pre_dim, argmax_dim, post_dim).astype(dtype)

        assert (argmax(arr, axis=1) == np.argmax(arr, axis=1)).all()


def test_argmax_no_post_dim(pre_dim, argmax_dim, dtype):
    for _ in range(N_STRESS):
        arr = np.random.randn(pre_dim, argmax_dim).astype(dtype)

        assert (argmax(arr, axis=-1) == np.argmax(arr, axis=-1)).all()


def test_argmax_no_pre_dim(argmax_dim, post_dim, dtype):
    for _ in range(N_STRESS):
        arr = np.random.randn(argmax_dim, post_dim).astype(dtype)

        assert (argmax(arr, axis=0) == np.argmax(arr, axis=0)).all()


def test_argmax_large_dim(dtype):
    arr = np.random.randn(134, 512, 123).astype(dtype)

    with pytest.warns(UserWarning):
        out = argmax(arr, axis=1)

    out_base = np.argmax(arr, axis=1)

    assert (out == out_base).all()


def test_argmax_not_c_contiguous(dtype):
    arr = np.random.randn(134, 127, 123).astype(dtype)
    arr = np.transpose(arr, (2, 1, 0))

    with pytest.warns(UserWarning):
        out = argmax(arr, axis=1)

    out_base = np.argmax(arr, axis=1)

    assert (out == out_base).all()


def test_argmax_bad_dtype():
    arr = np.random.randn(134, 127, 123).astype(bool)

    with pytest.warns(UserWarning):
        out = argmax(arr, axis=1)

    out_base = np.argmax(arr, axis=1)

    assert (out == out_base).all()
