from functools import partial
from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose
from scipy.ndimage import zoom as scipy_zoom

from imops.utils import get_c_contiguous_permutaion, inverse_permutation
from imops.zoom import _zoom, zoom, zoom_to_shape


# [:-1, :-1, :-1] below is used because of the strange scipy.ndimage.zoom behaviour at the edge
# https://github.com/scipy/scipy/issues/4922


# FIXME: fix inconsistency
# rtol=1e-6 as there is still some inconsistency
allclose = partial(allclose, rtol=1e-6)


@pytest.fixture(params=[False, True])
def fast(request):
    return request.param


@pytest.fixture(params=['scipy', 'cython', 'numba'])
def backend(request):
    return request.param


def test_shape(fast, backend):
    inp = np.random.rand(3, 10, 10) * 2 + 3
    shape = inp.shape

    assert zoom_to_shape(inp, shape, fast=fast, backend=backend).shape == shape
    assert zoom_to_shape(inp, shape[::-1], fast=fast, backend=backend).shape == shape[::-1]
    assert zoom(inp, (3, 4, 15), fast=fast, backend=backend).shape == (9, 40, 150)
    assert zoom(inp, (4, 3), axis=(1, 2), fast=fast, backend=backend).shape == (3, 40, 30)


def test_identity(fast, backend):
    for i in range(16):
        shape = np.random.randint(2, 128, size=np.random.randint(1, 4))
        inp = np.random.randn(*shape)

        allclose(inp, zoom(inp, 1, fast=fast, backend=backend), err_msg=f'{i, shape}')


def test_dtype(fast, backend):
    for inp_dtype in (np.float32, np.float64):
        for scale_dtype in (np.int32, np.float32, np.float64):
            for i in range(4):
                shape = np.random.randint(2, 128, size=np.random.randint(1, 4))
                inp = (10 * np.random.randn(*shape)).astype(inp_dtype)
                inp_copy = np.copy(inp)
                scale = np.random.uniform(0.5, 2, size=inp.ndim).astype(scale_dtype)

                without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]

                out = zoom(inp, scale, fast=fast, backend=backend)
                desired_out = scipy_zoom(inp, scale, order=1)

                allclose(out[without_borders], desired_out[without_borders], err_msg=f'{i, inp_dtype, scale_dtype}')
                assert (
                    out.dtype == desired_out.dtype == inp_dtype
                ), f'{i, out.dtype, desired_out.dtype, inp_dtype, scale_dtype}'

                allclose(inp, inp_copy, err_msg=f'{i, inp_dtype, scale_dtype}')
                assert (
                    inp.dtype == inp_copy.dtype == inp_dtype
                ), f'{i, inp.dtype, inp_copy.dtype, inp_dtype, scale_dtype}'


def test_scale_types(fast, backend):
    scales = [2, 2.0, (2, 2, 2), [2, 2, 2], np.array([2, 2, 2])]

    inp = np.random.randn(64, 64, 64)
    prev = None

    for scale in scales:
        out = zoom(inp, scale, fast=fast, backend=backend)

        if prev is not None:
            allclose(prev, out, err_msg=f'{scale}')

        prev = out


def test_contiguity_awareness(fast, backend):
    for i in range(2):
        for j in range(2):
            inp = np.random.randn(*(64,) * (3 - i))
            scale = np.random.uniform(0.5, 2, size=inp.ndim)

            zoom(inp, scale, fast=fast, backend=backend)

            desired_out = scipy_zoom(inp, scale, order=1)
            without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]

            for permutation in permutations(range(inp.ndim)):
                # This changes contiguity
                permuted = np.transpose(inp, permutation)

                out_permuted = zoom(permuted, scale[np.array(permutation)], backend=backend)

                allclose(
                    np.transpose(out_permuted, inverse_permutation(np.array(permutation)))[without_borders],
                    desired_out[without_borders],
                    err_msg=f'{i, j, permutation}',
                )

                assert (
                    get_c_contiguous_permutaion(permuted) is not None
                ), f"Didn't find permutation for {i, j, permutation}"


def test_thin(fast, backend):
    for i in range(3):
        for j in range(16):
            shape = [1 if k < i else np.random.randint(2, 128) for k in range(3)]
            inp = np.random.randn(*shape)
            scale = np.random.uniform(0.5, 2, size=3)

            without_borders = np.index_exp[
                : None if shape[0] == 1 else -1, : None if shape[1] == 1 else -1, : None if shape[2] == 1 else -1
            ]
            allclose(
                zoom(inp, scale, fast=fast, backend=backend)[without_borders],
                scipy_zoom(inp, scale, order=1)[without_borders],
                err_msg=f'{i, j, shape, scale}',
            )


def test_stress(fast, backend):
    """Make sure that our zoom-s are consistent with scipy's"""
    for i in range(32):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))
        inp = np.random.randn(*shape)
        scale = np.random.uniform(0.5, 2, size=inp.ndim if np.random.binomial(1, 0.5) else 1)
        if len(scale) == 1:
            scale = scale[0]

        without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]
        desired_out = scipy_zoom(inp, scale, order=1)[without_borders]

        allclose(
            zoom(inp, scale, fast=fast, backend=backend)[without_borders],
            desired_out,
            err_msg=f'{i, shape, scale}',
        )
        allclose(
            _zoom(inp, scale, fast=fast, backend=backend)[without_borders],
            desired_out,
            err_msg=f'{i, shape, scale}',
        )
