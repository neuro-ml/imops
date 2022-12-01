from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest
from utils import fill_outside, sample_ct, sk_iradon, sk_radon

from imops import inverse_radon, radon
from imops.backend import Backend, Cython


almost_eq = np.testing.assert_array_almost_equal

cython_configurations = [Cython(fast) for fast in [False, True]]
all_configurations = cython_configurations


@dataclass
class Alien5(Backend):
    pass


@pytest.fixture(params=all_configurations, ids=map(str, all_configurations))
def backend(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien5(), 'Alien6'], ids=['empty', 'Alien5', 'Alien6'])
def test_alien_backend(alien_backend):
    image = sample_ct(4, 64)
    sinogram = sk_radon(image)

    with pytest.raises(ValueError):
        radon(image, axes=(1, 2), backend=backend)

    with pytest.raises(ValueError):
        inverse_radon(sinogram, axes=(1, 2), backend=alien_backend)


def test_inverse_radon(backend):
    for slices in [1, 4]:
        for size in [40, 47, 64]:
            image = sample_ct(slices, size)
            sinogram = sk_radon(image)

            inv = inverse_radon(sinogram, axes=(1, 2), backend=backend)
            almost_eq(sk_iradon(sinogram), inv, 3)

            almost_eq(inv[:2], inverse_radon(sinogram[:2], axes=(1, 2), backend=backend))
            almost_eq(inv[[0]], inverse_radon(sinogram[[0]], axes=(1, 2), backend=backend))
            almost_eq(inv[0], inverse_radon(sinogram[0]))

            almost_eq(sk_iradon(sinogram[[0]]), inverse_radon(sinogram[[0]], axes=(1, 2), backend=backend), 3)
            almost_eq(inv, np.stack(list(map(partial(inverse_radon, backend=backend), sinogram))), 3)


def test_radon(backend):
    for slices in [1, 4]:
        for size in [40, 47, 64]:
            image = sample_ct(slices, size)

            sinogram = radon(image, axes=(1, 2), backend=backend)
            almost_eq(sk_radon(image), sinogram)
            almost_eq(sinogram, radon(fill_outside(image, -1000), axes=(1, 2), backend=backend))

            almost_eq(sinogram[:2], radon(image[:2], axes=(1, 2), backend=backend))
            almost_eq(sinogram[[0]], radon(image[[0]], axes=(1, 2), backend=backend))
            almost_eq(sinogram[0], radon(image[0], backend=backend))

            almost_eq(sk_radon(image[[0]]), radon(image[[0]], axes=(1, 2), backend=backend), 3)
            almost_eq(sinogram, np.stack(list(map(partial(radon, backend=backend), image))), 3)
