from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose
from scipy.ndimage import generate_binary_structure
from skimage.morphology import binary_dilation as sk_binary_dilation

from imops import binary_dilation
from imops.backend import Backend, Cython


allclose = partial(allclose, rtol=1e-6)
cython_configurations = [Cython(fast) for fast in [False, True]]

names = list(map(str, cython_configurations))


@dataclass
class Alien1(Backend):
    pass


@pytest.fixture(params=cython_configurations, ids=names)
def backend(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien1(), 'Alien2'], ids=['empty', 'Alien1', 'Alien2'])
def test_alien_backend(alien_backend):
    inp = np.random.binomial(1, 0.5, (32, 32))

    with pytest.raises(ValueError):
        binary_dilation(inp, backend=alien_backend)


def test_stress(backend):
    for i in range(32):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))
        inp = np.random.binomial(1, 0.5, shape)
        footprint = generate_binary_structure(inp.ndim, np.random.randint(1, 3))

        desired_out = sk_binary_dilation(inp, footprint)

        allclose(
            binary_dilation(inp, footprint, backend=backend),
            desired_out,
            err_msg=f'{i, shape, footprint}',
        )
