from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose
from scipy.ndimage import rotate as scipy_rotate

from imops._configs import rotate_configs
from imops.backend import Backend
from imops.rotate import DTYPES, rotate


np.random.seed(1337)

allclose = partial(allclose, rtol=1e-6, atol=1e-6)
ANGLES = [0, 1.5, 30, 45, 90, 180, 270, 359.5, -30, 400]
PLANES = [(0, 1), (0, 2), (1, 2), (-1, -2)]


@dataclass
class Alien11(Backend):
    pass


@pytest.fixture(params=rotate_configs, ids=map(str, rotate_configs))
def backend(request):
    return request.param


@pytest.fixture(params=[0, 1])
def order(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['reshape', 'keep_shape'])
def reshape(request):
    return request.param


def cube_with_ball(shape):
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, size) for size in shape], indexing='ij')

    return 37 * (np.maximum(np.maximum(abs(z), abs(y)), abs(x)) <= 0.6) + 74 * (z**2 + y**2 + x**2 <= 0.3)


@pytest.mark.parametrize('alien_backend', ['', Alien11(), 'Alien12'], ids=['empty', 'Alien11', 'Alien12'])
def test_alien_backend(alien_backend):
    with pytest.raises(ValueError):
        rotate(np.random.randn(8, 8, 8).astype('float32'), 30, backend=alien_backend)


def test_invalid_plane():
    inp = np.random.randn(8, 8, 8).astype('float32')

    for axes in [(0,), (0, 1, 2), (1, 1), (0, 3), (0, -4)]:
        with pytest.raises(ValueError):
            rotate(inp, 30, axes=axes)

    with pytest.raises(ValueError):
        rotate(np.random.randn(8).astype('float32'), 30)


def test_single_threaded_warning():
    with pytest.warns(UserWarning):
        rotate(np.random.randn(8, 8, 8).astype('float32'), 30, num_threads=2, backend='Scipy')


def test_fallback_warning():
    inp = np.random.randn(8, 8, 8).astype('float32')

    with pytest.warns(UserWarning):
        rotate(inp, 30, order=3)

    with pytest.warns(UserWarning):
        rotate(inp.astype('float64'), 30)


def test_shape(backend, order, reshape):
    inp = np.random.randn(7, 16, 24).astype('float32')

    for angle, axes in product(ANGLES, PLANES):
        out = rotate(inp, angle, axes=axes, reshape=reshape, order=order, backend=backend)
        desired = scipy_rotate(inp, angle, axes=axes, reshape=reshape, order=order)

        assert out.shape == desired.shape, f'{angle, axes}'


def test_against_scipy(backend, order, reshape):
    inp = cube_with_ball((15, 32, 29)).astype('float32')

    for angle, axes in product(ANGLES, PLANES):
        allclose(
            rotate(inp, angle, axes=axes, reshape=reshape, order=order, backend=backend),
            scipy_rotate(inp, angle, axes=axes, reshape=reshape, order=order),
            err_msg=f'{angle, axes}',
        )


def test_dtype(backend, order, reshape):
    for dtype in DTYPES:
        inp = cube_with_ball((9, 24, 20)).astype(dtype)
        inp_copy = inp.copy()

        out = rotate(inp, 37, axes=(1, 2), reshape=reshape, order=order, backend=backend)
        desired = scipy_rotate(inp, 37, axes=(1, 2), reshape=reshape, order=order)

        allclose(out, desired, err_msg=f'{dtype}')
        assert out.dtype == desired.dtype == dtype, f'{dtype, out.dtype, desired.dtype}'
        allclose(inp, inp_copy, err_msg=f'{dtype}')


def test_integer_rounding(backend, order, reshape):
    for dtype in [np.uint8, np.uint16, np.int16, np.int32]:
        low = 0 if np.iinfo(dtype).min == 0 else -97
        inp = np.random.randint(low, 97, (6, 20, 18)).astype(dtype)

        for cval in [-3.7, 0.0, 5.5]:
            out = rotate(inp, 23, axes=(1, 2), reshape=reshape, order=order, cval=cval, backend=backend)
            desired = scipy_rotate(inp, 23, axes=(1, 2), reshape=reshape, order=order, cval=cval)

            assert np.array_equal(out, desired), f'{dtype, cval}'


def test_float16(backend, order, reshape):
    inp = cube_with_ball((9, 24, 20)).astype(np.float16)

    out = rotate(inp, 37, axes=(1, 2), reshape=reshape, order=order, backend=backend)
    desired = scipy_rotate(inp.astype(np.float32), 37, axes=(1, 2), reshape=reshape, order=order)

    assert out.dtype == np.float16
    allclose(out, desired.astype(np.float16))


def test_ndim(backend, order, reshape):
    for shape, axes in [((16, 20), (0, 1)), ((5, 16, 20), (1, 2)), ((2, 5, 16, 20), (2, 3)), ((2, 3, 4, 5, 6), (1, 3))]:
        inp = np.random.randn(*shape).astype('float32')

        allclose(
            rotate(inp, 33, axes=axes, reshape=reshape, order=order, backend=backend),
            scipy_rotate(inp, 33, axes=axes, reshape=reshape, order=order),
            err_msg=f'{shape, axes}',
        )


def test_contiguous_output(backend, order, reshape):
    for shape, axes in [((16, 20), (0, 1)), ((9, 16, 20), (0, 1)), ((9, 16, 20), (0, 2)), ((9, 16, 20), (1, 2))]:
        inp = np.random.randn(*shape).astype('float32')
        out = rotate(inp, 33, axes=axes, reshape=reshape, order=order, backend=backend)

        assert out.flags.c_contiguous, f'{shape, axes}'


def test_cval(backend, order, reshape):
    inp = cube_with_ball((9, 24, 20)).astype('float32')

    for cval in [-5.0, 0.0, 7.5]:
        allclose(
            rotate(inp, 41, axes=(1, 2), reshape=reshape, order=order, cval=cval, backend=backend),
            scipy_rotate(inp, 41, axes=(1, 2), reshape=reshape, order=order, cval=cval),
            err_msg=f'{cval}',
        )


def test_noncontiguous(backend, order, reshape):
    inp = cube_with_ball((18, 24, 20)).astype('float32')[::2].T

    allclose(
        rotate(inp, 29, axes=(0, 1), reshape=reshape, order=order, backend=backend),
        scipy_rotate(inp, 29, axes=(0, 1), reshape=reshape, order=order),
    )


def test_num_threads(backend, order):
    inp = cube_with_ball((12, 40, 40)).astype('float32')
    single = rotate(inp, 23, axes=(1, 2), order=order, num_threads=1, backend=backend)

    for num_threads in [2, 4, -1]:
        allclose(rotate(inp, 23, axes=(1, 2), order=order, num_threads=num_threads, backend=backend), single)
