import numpy as np
import pytest

from imops.argmax import argmax


N_STRESS = 1000
np.random.seed(1337)
dim = 13


@pytest.fixture(params=[1, 2, 3, 54, 72, 128, 256])
def pre_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 54, 72, 128, 256])
def post_dim(request):
    return request.param


@pytest.fixture(params=[2, 3, 5, 7, 11, 18])
def argmax_dim(request):
    return request.param


def test_argmax(pre_dim, argmax_dim, post_dim):
    for _ in range(N_STRESS):
        arr = np.random.randn(pre_dim, argmax_dim, post_dim).astype(np.float32)

        assert (argmax(arr, axis=1) == np.argmax(arr, axis=1)).all()


def test_argmax_no_post_dim(pre_dim, argmax_dim):
    for _ in range(N_STRESS):
        arr = np.random.randn(pre_dim, argmax_dim).astype(np.float32)

        assert (argmax(arr, axis=-1) == np.argmax(arr, axis=-1)).all()


def test_argmax_no_pre_dim(argmax_dim, post_dim):
    for _ in range(N_STRESS):
        arr = np.random.randn(argmax_dim, post_dim).astype(np.float32)

        assert (argmax(arr, axis=0) == np.argmax(arr, axis=0)).all()


def test_argmax_large_dim():
    arr = np.random.randn(134, 512, 123).astype(np.float32)

    with pytest.warns(UserWarning):
        out = argmax(arr, axis=1)

    out_base = np.argmax(arr, axis=1)

    assert (out == out_base).all()
