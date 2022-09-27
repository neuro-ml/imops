import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose
from scipy.interpolate import interp1d as scipy_interp1d

from imops import interp1d


@pytest.fixture(params=[False, True])
def fast(request):
    return request.param


@pytest.fixture(params=['scipy', 'cython', 'numba'])
def backend(request):
    return request.param


def test_extrapolation_exception(fast, backend):
    x = np.array([1.0, 2.0, 3.0])
    x_new = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        scipy_interp1d(x, y, axis=0, fill_value=0)(x_new)
    with pytest.raises(ValueError):
        interp1d(x, y, axis=0, fill_value=0, fast=fast, backend=backend)(x_new)


def test_length_inequality_exception(fast, backend):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.random.randn(3, 4, 5)

    with pytest.raises(ValueError):
        scipy_interp1d(x, y)
    with pytest.raises(ValueError):
        interp1d(x, y, fast=fast, backend=backend)

    with pytest.raises(ValueError):
        scipy_interp1d(x, y, axis=2)
    with pytest.raises(ValueError):
        interp1d(x, y, axis=2, fast=fast, backend=backend)

    with pytest.raises(ValueError):
        scipy_interp1d(x, y, axis=0)
    with pytest.raises(ValueError):
        interp1d(x, y, axis=0, fast=fast, backend=backend)


def test_extrapolation(fast, backend):
    for i in range(16):
        shape = np.random.randint(16, 64, size=np.random.randint(1, 4))
        inp = np.random.randn(*shape)

        axis = np.random.choice(np.arange(inp.ndim))

        extrapolation = False
        while not extrapolation:
            old_locations = np.random.randn(shape[axis])
            new_locations = np.random.randn(np.random.randint(shape[axis] // 2, shape[axis] * 2)) * 2

            extrapolation = np.max(new_locations) > np.max(old_locations) or np.min(new_locations) < np.min(
                old_locations
            )

        out = interp1d(
            old_locations, inp, axis=axis, bounds_error=False, fill_value='extrapolate', fast=fast, backend=backend
        )(new_locations)
        desired_out = scipy_interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value='extrapolate')(
            new_locations
        )

        allclose(out, desired_out, err_msg=f'{i, shape}')


def test_dtype(fast, backend):
    for inp_dtype in (np.float32, np.float64):
        for old_locations_dtype in (np.float32, np.float64):
            for new_locations_dtype in (np.float32, np.float64):
                shape = (128, 128)
                inp = np.random.randn(*shape).astype(inp_dtype)

                axis = np.random.choice(np.arange(inp.ndim))
                old_locations = np.random.randn(shape[axis]).astype(old_locations_dtype)
                new_locations = np.random.randn(np.random.randint(shape[axis] // 2, shape[axis] * 2)).astype(
                    new_locations_dtype
                )

                out = interp1d(
                    old_locations, inp, axis=axis, bounds_error=False, fill_value=0, fast=fast, backend=backend
                )(new_locations)

                assert out.dtype == max(
                    inp_dtype, old_locations_dtype, new_locations_dtype, key=lambda x: x(0).itemsize
                ), f'{out.dtype}, {inp_dtype}, {old_locations_dtype}, {new_locations_dtype}'


def test_stress(fast, backend):
    for i in range(64):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
        inp = np.random.randn(*shape)

        axis = np.random.choice(np.arange(inp.ndim))
        old_locations = np.random.randn(shape[axis])
        new_locations = np.random.randn(np.random.randint(shape[axis] // 2, shape[axis] * 2))

        out = interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value=0, fast=fast, backend=backend)(
            new_locations
        )
        desired_out = scipy_interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value=0)(new_locations)

        allclose(out, desired_out, err_msg=f'{i, shape}')
