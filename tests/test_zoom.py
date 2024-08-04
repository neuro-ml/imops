from dataclasses import dataclass
from functools import partial
from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose
from scipy.ndimage import zoom as scipy_zoom

from imops._configs import zoom_configs
from imops.backend import Backend
from imops.utils import ZOOM_SRC_DIM, get_c_contiguous_permutaion, inverse_permutation, make_immutable
from imops.zoom import _zoom, zoom, zoom_to_shape


# [:-1, :-1, :-1, :-1, :-1] below is used because of the strange scipy.ndimage.zoom behaviour at the edge
# https://github.com/scipy/scipy/issues/4922

np.random.seed(1337)

# FIXME: fix inconsistency
# rtol=1e-6 as there is still some inconsistency
allclose = partial(allclose, rtol=1e-6)
n_samples = 8


@dataclass
class Alien1(Backend):
    pass


@pytest.fixture(params=zoom_configs, ids=map(str, zoom_configs))
def backend(request):
    return request.param


@pytest.fixture(params=[0, 1])
def order(request):
    return request.param


@pytest.fixture(params=[np.float32, np.float64, bool, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32])
def dtype(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien1(), 'Alien2'], ids=['empty', 'Alien1', 'Alien2'])
def test_alien_backend(alien_backend):
    inp = np.random.randn(32, 32, 32)

    with pytest.raises(ValueError):
        zoom(inp, 2, backend=alien_backend)


def test_single_threaded_warning(order):
    inp = np.random.randn(32, 32, 32)

    with pytest.warns(UserWarning):
        zoom(inp, 2, order=order, num_threads=2, backend='Scipy')


def test_numba_num_threads(order):
    inp = np.random.randn(32, 32, 32)

    with pytest.warns(UserWarning):
        zoom(inp, 2, order=order, num_threads=2, backend='Numba')


def test_callable_fill_value(backend, order):
    inp = np.random.randn(64, 64, 64)
    scale = np.random.uniform(0.5, 1.5, size=inp.ndim)

    out = zoom(inp, scale, fill_value=np.min, order=order, backend=backend)
    without_borders = np.index_exp[:-1, :-1, :-1]

    allclose(
        scipy_zoom(inp, scale, cval=np.min(inp), order=order)[without_borders], out[without_borders], err_msg=f'{scale}'
    )


def test_shape(backend, order):
    inp = np.random.rand(3, 10, 10) * 2 + 3
    shape = inp.shape

    assert zoom_to_shape(inp, shape, order=order, backend=backend).shape == shape
    assert zoom_to_shape(inp, shape[::-1], order=order, backend=backend).shape == shape[::-1]
    assert zoom(inp, (3, 4, 15), order=order, backend=backend).shape == (9, 40, 150)
    assert zoom(inp, (4, 3), axis=(1, 2), order=order, backend=backend).shape == (3, 40, 30)


def test_identity(backend, order, dtype):
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    for i in range(n_samples):
        shape = np.random.randint(2, 32, size=np.random.randint(1, ZOOM_SRC_DIM + 1))
        inp = np.random.randn(*shape)
        if dtype == bool:
            inp = inp > 0
        else:
            inp = inp.astype(dtype)

        allclose(inp, zoom(inp, 1, order=order, backend=backend), err_msg=f'{i, shape}')


def test_dtype(backend, order, dtype):
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    for scale_dtype in (np.int32, np.float32, np.float64):
        for i in range(n_samples):
            shape = np.random.randint(2, 32, size=np.random.randint(1, ZOOM_SRC_DIM + 1))
            inp = np.random.randn(*shape)
            if dtype == bool:
                inp = inp > 0
            else:
                inp = inp.astype(dtype)
            inp_copy = np.copy(inp)
            scale = np.random.uniform(0.5, 1.5, size=inp.ndim).astype(scale_dtype)

            without_borders = np.index_exp[:-1, :-1, :-1, :-1, :-1][: inp.ndim]

            out = zoom(inp, scale, order=order, backend=backend)
            desired_out = scipy_zoom(inp, scale, order=order)

            allclose(out[without_borders], desired_out[without_borders], err_msg=f'{i, dtype, scale_dtype}')
            assert out.dtype == desired_out.dtype == dtype, f'{i, out.dtype, desired_out.dtype, dtype, scale_dtype}'

            allclose(inp, inp_copy, err_msg=f'{i, dtype, scale_dtype}')
            assert inp.dtype == inp_copy.dtype == dtype, f'{i, inp.dtype, inp_copy.dtype, dtype, scale_dtype}'


def test_scale_types(backend, order, dtype):
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    scales = [2, 2.0, (2, 2, 2), [2, 2, 2], np.array([2, 2, 2])]

    inp = np.random.randn(64, 64, 64)
    if dtype == bool:
        inp = inp > 0
    else:
        inp = inp.astype(dtype)
    prev = None

    for scale in scales:
        out = zoom(inp, scale, order=order, backend=backend)

        if prev is not None:
            allclose(prev, out, err_msg=f'{scale}')

        prev = out


def test_contiguity_awareness(backend, order, dtype):
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    for i in range(ZOOM_SRC_DIM):
        inp = np.random.randn(*(32,) * (ZOOM_SRC_DIM - i))
        if dtype == bool:
            inp = inp > 0
        else:
            inp = inp.astype(dtype)

        scale = np.random.uniform(0.5, 1.5, size=inp.ndim)

        zoom(inp, scale, order=order, backend=backend)

        desired_out = scipy_zoom(inp, scale, order=order)
        without_borders = np.index_exp[:-1, :-1, :-1, :-1, :-1][: inp.ndim]

        for permutation in permutations(range(inp.ndim)):
            # This changes contiguity
            permuted = np.transpose(inp, permutation)

            out_permuted = zoom(permuted, scale[np.array(permutation)], order=order, backend=backend)

            allclose(
                np.transpose(out_permuted, inverse_permutation(np.array(permutation)))[without_borders],
                desired_out[without_borders],
                err_msg=f'{i, permutation}',
            )

            assert get_c_contiguous_permutaion(permuted) is not None, f"Didn't find permutation for {i, permutation}"


def test_nocontiguous(backend, order, dtype):
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    inp = np.random.randn(64, 64, 64)
    if dtype == bool:
        inp = inp > 0
    else:
        inp = inp.astype(dtype)
    inp = inp[::2]

    scale = 2

    desired_out = scipy_zoom(inp, scale, order=order)
    without_borders = np.index_exp[:-1, :-1, :-1][: inp.ndim]

    if backend.name == 'Scipy':
        allclose(zoom(inp, 2, order=order, backend=backend)[without_borders], desired_out[without_borders])
    else:
        with pytest.warns(UserWarning):
            allclose(zoom(inp, 2, order=order, backend=backend)[without_borders], desired_out[without_borders])


def test_thin(backend, order, dtype):
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    for i in range(ZOOM_SRC_DIM):
        for j in range(n_samples):
            shape = [1 if k < i else np.random.randint(2, 32) for k in range(ZOOM_SRC_DIM + 1)]
            inp = np.random.randn(*shape)
            if dtype == bool:
                inp = inp > 0
            else:
                inp = inp.astype(dtype)
            scale = np.random.uniform(0.5, 1.5, size=len(shape))

            without_borders = np.index_exp[
                : None if shape[0] == 1 else -1,
                : None if shape[1] == 1 else -1,
                : None if shape[2] == 1 else -1,
                : None if shape[3] == 1 else -1,
            ]
            allclose(
                zoom(inp, scale, order=order, backend=backend)[without_borders],
                scipy_zoom(inp, scale, order=order)[without_borders],
                err_msg=f'{i, j, shape, scale}',
            )


def test_stress(backend, order, dtype):
    """Make sure that our zoom-s are consistent with scipy's"""
    if order == 1 and dtype not in (np.float32, np.float64):
        return

    for i in range(n_samples):
        shape = np.random.randint(16, 32, size=np.random.randint(1, ZOOM_SRC_DIM + 1))
        inp = np.random.randn(*shape)
        if dtype == bool:
            inp = inp > 0
        else:
            inp = inp.astype(dtype)
        scale = np.random.uniform(0.5, 2, size=inp.ndim if np.random.binomial(1, 0.5) else 1)
        if len(scale) == 1:
            scale = scale[0]
        if np.random.binomial(1, 0.5):
            make_immutable(inp)

        without_borders = np.index_exp[:-1, :-1, :-1, :-1, :-1][: inp.ndim]
        desired_out = scipy_zoom(inp, scale, order=order)[without_borders]

        allclose(
            zoom(inp, scale, order=order, backend=backend)[without_borders],
            desired_out,
            err_msg=f'{i, shape, scale}',
        )
        allclose(
            _zoom(inp, scale, order=order, backend=backend)[without_borders],
            desired_out,
            err_msg=f'{i, shape, scale}',
        )
