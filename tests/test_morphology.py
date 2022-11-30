from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import generate_binary_structure
from skimage.morphology import (
    binary_closing as sk_binary_closing,
    binary_dilation as sk_binary_dilation,
    binary_erosion as sk_binary_erosion,
    binary_opening as sk_binary_opening,
)

from imops.backend import Backend, Cython, Scipy
from imops.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening


scipy_configurations = [Scipy()]
cython_configurations = [Cython(fast) for fast in [False, True]]
all_configurations = scipy_configurations + cython_configurations

all_configurations_names = list(map(str, all_configurations))

test_pairs = [
    [sk_binary_dilation, binary_dilation],
    [sk_binary_erosion, binary_erosion],
    [sk_binary_closing, binary_closing],
    [sk_binary_opening, binary_opening],
]
test_pairs_names = [x[0].__name__ for x in test_pairs]


@dataclass
class Alien1(Backend):
    pass


@pytest.fixture(params=all_configurations, ids=all_configurations_names)
def backend(request):
    return request.param


@pytest.fixture(params=test_pairs, ids=test_pairs_names)
def test_pair(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien1(), 'Alien2'], ids=['empty', 'Alien1', 'Alien2'])
def test_alien_backend(alien_backend):
    inp = np.random.binomial(1, 0.5, (32, 32))

    with pytest.raises(ValueError):
        binary_dilation(inp, backend=alien_backend)


def test_stress(test_pair, backend):
    sk_op, imops_op = test_pair

    for i in range(32):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))
        inp = np.random.binomial(1, 0.5, shape)
        footprint = generate_binary_structure(inp.ndim, np.random.randint(1, 3))

        desired_out = sk_op(inp, footprint)

        assert_array_equal(
            imops_op(inp, footprint, backend=backend),
            desired_out,
            err_msg=f'{i, shape, footprint}',
        )
