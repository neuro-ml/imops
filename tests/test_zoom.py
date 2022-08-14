from itertools import permutations
# from time import time

import numpy as np
import pytest
from dpipe.im import zoom as dpipe_zoom, zoom_to_shape as dpipe_zoom_to_shape
from numpy.testing import assert_allclose as allclose
from scipy.ndimage import zoom as scipy_zoom

from imops import _zoom, zoom, zoom_to_shape
from imops.utils import get_c_contiguous_permutaion, inverse_permutation

# [:-1, :-1, :-1] below is used because of the strange scipy.ndimage.zoom behaviour at the edge
# https://github.com/scipy/scipy/issues/4922
SEED = 1337


@pytest.fixture(params=[(_zoom, scipy_zoom), (zoom, dpipe_zoom)])
def zooms(request):
    return request.param


def test_zoom_to_shape():
    for i in range(16):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))
        new_shape = np.random.poisson(shape) + 1
        inp = np.random.randn(*shape)
        axis = np.arange(len(inp.shape))

        without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]

        # rtol=1e-6 as there is still some inconsistency
        allclose(
            zoom_to_shape(inp, new_shape, axis=axis)[without_borders],
            dpipe_zoom_to_shape(inp, new_shape, order=1, axis=axis)[without_borders],
            rtol=1e-6,
            err_msg=f'{i, shape, new_shape}',
        )


def test_identity(zooms):
    test_zoom, _ = zooms
    for i in range(16):
        shape = np.random.randint(2, 128, size=np.random.randint(1, 4))
        inp = np.random.randn(*shape)

        allclose(inp, test_zoom(inp, np.ones(inp.ndim)), err_msg=f'{i, shape}')


def test_dtype(zooms):
    test_zoom, target_zoom = zooms
    for dtype in (np.float32, np.float64):
        for i in range(4):
            shape = np.random.randint(2, 128, size=np.random.randint(1, 4))
            inp = (10 * np.random.randn(*shape)).astype(dtype)
            inp_copy = np.copy(inp)
            scale = np.random.uniform(0.5, 2, size=1 if np.random.binomial(1, 0.5) else inp.ndim)
            if len(scale) == 1:
                scale = scale[0]

            without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]
            axis = {'axis': np.arange(len(shape))} if test_zoom == zoom else {}

            out = test_zoom(inp, scale, **axis)
            desired_out = target_zoom(inp, scale, order=1, **axis)

            # FIXME: fix inconsistency
            # rtol=1e-6 as there is still some inconsistency
            allclose(out[without_borders], desired_out[without_borders], err_msg=f'{i, dtype}', rtol=1e-6)
            assert out.dtype == desired_out.dtype == dtype, f'{i, out.dtype, desired_out.dtype, dtype}'

            allclose(inp, inp_copy, err_msg=f'{i, dtype}', rtol=1e-6)
            assert inp.dtype == inp_copy.dtype == dtype, f'{i, inp.dtype, inp_copy.dtype, dtype}'


def test_scale_types(zooms):
    test_zoom, target_zoom = zooms
    scales = [2, 2.0, (2, 2, 2), [2, 2, 2], np.array([2, 2, 2])]

    inp = np.random.randn(64, 64, 64)
    axis = {'axis': np.arange(len(inp.shape))} if test_zoom == zoom else {}

    prev = None

    for scale in scales:
        out = test_zoom(inp, scale, **axis)
        desired_out = target_zoom(inp, scale, order=1, **axis)

        allclose(out, desired_out, err_msg=f'{scale}')

        if prev is not None:
            allclose(prev, out, err_msg=f'{scale}')

        prev = out


def test_contiguity_awareness(zooms):
    test_zoom, target_zoom = zooms
    for i in range(2):
        for j in range(2):
            inp = np.random.randn(*(64,) * (3 - i))
            scale = np.random.uniform(0.5, 2, size=inp.ndim)
            axis = {'axis': np.arange(len(inp.shape))} if test_zoom == zoom else {}

            # start = time()
            test_zoom(inp, scale, **axis)
            # runtime = time() - start

            desired_out = target_zoom(inp, scale, order=1, **axis)
            without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]

            for permutation in permutations(range(inp.ndim)):
                # This changes contiguity
                permuted = np.transpose(inp, permutation)

                # start_permuted = time()
                out_permuted = test_zoom(permuted, scale[np.array(permutation)], **axis)
                # runtime_permuted = time() - start_permuted

                allclose(
                    np.transpose(out_permuted, inverse_permutation(np.array(permutation)))[without_borders],
                    desired_out[without_borders],
                    err_msg=f'{i, j, permutation}',
                )

                # This might not pass time to time
                # allclose(runtime_permuted, runtime, rtol=1 if runtime > 0.1 else 10, err_msg=f'{i, j, permutation}')
                assert (
                    get_c_contiguous_permutaion(permuted) is not None
                ), f"Didn't find permutation for {i, j, permutation}"


def test_thin(zooms):
    test_zoom, target_zoom = zooms
    for i in range(3):
        for j in range(16):
            shape = [1 if k < i else np.random.randint(2, 128) for k in range(3)]
            inp = np.random.randn(*shape)
            scale = np.random.uniform(0.5, 2, size=3)
            axis = {'axis': np.arange(len(inp.shape))} if test_zoom == zoom else {}

            without_borders = np.index_exp[
                : None if shape[0] == 1 else -1, : None if shape[1] == 1 else -1, : None if shape[2] == 1 else -1
            ]
            allclose(
                test_zoom(inp, scale, **axis)[without_borders],
                target_zoom(inp, scale, order=1, **axis)[without_borders],
                err_msg=f'{i, j, shape, scale}',
            )


def test_stress(zooms):
    test_zoom, target_zoom = zooms
    for i in range(32):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))
        inp = np.random.randn(*shape)
        scale = np.random.uniform(0.5, 2, size=1 if np.random.binomial(1, 0.5) else inp.ndim)
        if len(scale) == 1:
            scale = scale[0]
        axis = {'axis': np.arange(len(inp.shape))} if test_zoom == zoom else {}

        without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]

        # rtol=1e-6 as there is still some inconsistency
        allclose(
            test_zoom(inp, scale, **axis)[without_borders],
            target_zoom(inp, scale, order=1, **axis)[without_borders],
            rtol=1e-6,
            err_msg=f'{i, shape, scale}',
        )
