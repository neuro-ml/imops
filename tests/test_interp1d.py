from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose
from scipy.interpolate import interp1d as scipy_interp1d

from imops._configs import interp1d_configs
from imops.backend import Backend, Scipy
from imops.interp1d import interp1d
from imops.utils import make_immutable


np.random.seed(1337)
n_samples = 8


@dataclass
class Alien3(Backend):
    pass


@pytest.fixture(params=interp1d_configs, ids=map(str, interp1d_configs))
def backend(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien3(), 'Alien4'], ids=['empty', 'Alien3', 'Alien4'])
def test_alien_backend(alien_backend):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        interp1d(x, y, axis=0, fill_value=0, backend=alien_backend)


def test_single_threaded_warning():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.warns(UserWarning):
        interp1d(x, y, axis=0, fill_value=0, num_threads=2, backend='Scipy')(x)


def test_extrapolate_error(backend):
    x = np.array([0.0])
    y = np.array([1.0])

    if backend != Scipy():
        with pytest.raises(ValueError):
            interp1d(x, y, fill_value='extrapolate', backend=backend)

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        interp1d(x, y, fill_value='extrapolate', bounds_error=True, backend=backend)


def test_numba_num_threads():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.warns(UserWarning):
        interp1d(x, y, axis=0, fill_value=0, num_threads=2, backend='Numba')(x)


def test_extrapolation_exception(backend):
    x = np.array([1.0, 2.0, 3.0])
    x_new = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        scipy_interp1d(x, y, axis=0, fill_value=0)(x_new)
    with pytest.raises(ValueError):
        interp1d(x, y, axis=0, fill_value=0, backend=backend)(x_new)


def test_length_inequality_exception(backend):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.random.randn(3, 4, 5)

    with pytest.raises(ValueError):
        scipy_interp1d(x, y)
    with pytest.raises(ValueError):
        interp1d(x, y, backend=backend)

    with pytest.raises(ValueError):
        scipy_interp1d(x, y, axis=2)
    with pytest.raises(ValueError):
        interp1d(x, y, axis=2, backend=backend)

    with pytest.raises(ValueError):
        scipy_interp1d(x, y, axis=0)
    with pytest.raises(ValueError):
        interp1d(x, y, axis=0, backend=backend)


def test_nans(backend):
    if backend.name == 'Scipy':
        return

    x = np.array([0, 1, 2])
    y = np.array([np.inf, -np.inf, np.inf])

    with pytest.raises(RuntimeError):
        interp1d(x, y, axis=0, fill_value=0, backend=backend)(x / 2)

    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([np.inf, 0, 1, 2, -np.inf, np.inf])

    with pytest.raises(RuntimeError):
        interp1d(x, y, axis=0, fill_value=0, backend=backend)(x)

    y = np.array([np.inf, 0, 1, np.inf, -np.inf, np.inf])

    allclose(
        interp1d(x, y, axis=0, fill_value=0, backend=backend)(x / 2),
        np.array([np.inf, np.inf, np.inf, 0.5, 1, np.inf]),
    )
    allclose(
        interp1d(x, -y, axis=0, fill_value=0, backend=backend)(x / 2),
        np.array([-np.inf, -np.inf, -np.inf, -0.5, -1, -np.inf]),
    )


def test_extrapolation(backend):
    for i in range(n_samples):
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

        out = interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value='extrapolate', backend=backend)(
            new_locations
        )
        desired_out = scipy_interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value='extrapolate')(
            new_locations
        )

        allclose(out, desired_out, err_msg=f'{i, shape}')


def test_dtype(backend):
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

                out = interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value=0, backend=backend)(
                    new_locations
                )

                assert out.dtype == max(
                    inp_dtype, old_locations_dtype, new_locations_dtype, key=lambda x: x(0).itemsize
                ), f'{out.dtype}, {inp_dtype}, {old_locations_dtype}, {new_locations_dtype}'


def test_stress(backend):
    for i in range(2 * n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
        inp = np.random.randn(*shape)
        if np.random.binomial(1, 0.5):
            make_immutable(inp)

        axis = np.random.choice(np.arange(inp.ndim))
        old_locations = np.random.randn(shape[axis])
        new_locations = np.random.randn(np.random.randint(shape[axis] // 2, shape[axis] * 2))

        out = interp1d(
            old_locations,
            inp,
            axis=axis,
            copy=np.random.binomial(1, 0.5),
            bounds_error=False,
            fill_value=0,
            backend=backend,
        )(new_locations)
        desired_out = scipy_interp1d(old_locations, inp, axis=axis, bounds_error=False, fill_value=0)(new_locations)

        allclose(out, desired_out, rtol=1e-6, err_msg=f'{i, shape}')
