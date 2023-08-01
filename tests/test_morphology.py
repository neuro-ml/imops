from dataclasses import dataclass

import numpy as np
import pytest
from skimage.morphology import (
    binary_closing as sk_binary_closing,
    binary_dilation as sk_binary_dilation,
    binary_erosion as sk_binary_erosion,
    binary_opening as sk_binary_opening,
)

from imops._configs import morphology_configs
from imops.backend import Backend, Scipy
from imops.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening
from imops.pad import restore_crop


np.random.seed(1337)

assert_eq = np.testing.assert_array_equal
test_pairs = [
    [sk_binary_erosion, binary_erosion],
    [sk_binary_dilation, binary_dilation],
    [sk_binary_opening, binary_opening],
    [sk_binary_closing, binary_closing],
]


@dataclass
class Alien7(Backend):
    pass


@pytest.fixture(params=[False, True], ids=['boxed=False', 'boxed=True'])
def boxed(request):
    return request.param


@pytest.fixture(params=morphology_configs, ids=map(str, morphology_configs))
def backend(request):
    return request.param


@pytest.fixture(params=test_pairs, ids=(x[0].__name__ for x in test_pairs))
def pair(request):
    return request.param


@pytest.fixture(
    params=[lambda x: 2 * x, lambda x: 2 * x - 1, lambda x: x],
    ids=['even shaped footprint', 'odd shaped footprint', 'random shaped footprint'],
)
def footprint_shape_modifier(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien7(), 'Alien8'], ids=['empty', 'Alien7', 'Alien8'])
def test_alien_backend(alien_backend):
    inp = np.random.binomial(1, 0.5, (32, 32))

    with pytest.raises(ValueError):
        binary_dilation(inp, backend=alien_backend)


def test_single_threaded_warning(pair):
    _, imops_op = pair
    with pytest.warns(UserWarning):
        imops_op(np.ones(1), num_threads=2, backend='Scipy')


def test_empty(pair, backend):
    _, imops_op = pair
    with pytest.raises(RuntimeError):
        imops_op(np.ones(1), np.array([]))


def test_stress(pair, backend, footprint_shape_modifier, boxed):
    # FIXME
    def take_by_coords(array, coords):
        copy_array = np.copy(array)
        for coord in coords:
            copy_array = copy_array[coord]

        return copy_array

    sk_op, imops_op = pair

    for i in range(32):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))

        if boxed:
            box_size = np.asarray([np.random.randint(s // 3, s + 1) for s in shape])
            box_pos = np.asarray([np.random.randint(0, s - bs + 1) for bs, s in zip(box_size, shape)])
            box_coord = np.array([box_pos, box_pos + box_size])
            inp = np.random.binomial(1, 0.7, box_size)
            inp = restore_crop(inp, box_coord, shape, 0)
        else:
            inp = np.random.binomial(1, 0.5, shape)

        footprint_shape = footprint_shape_modifier(np.random.randint(1, 4, size=inp.ndim))
        footprint = np.random.binomial(1, 0.5, footprint_shape) if np.random.binomial(1, 0.5, 1) else None

        if backend == Scipy() and boxed:
            with pytest.raises(ValueError):
                imops_op(inp, footprint, backend=backend, boxed=boxed)
            return

        if (
            boxed
            and footprint is not None
            and (
                ((np.asarray(footprint.shape) % 2) == 0).any()
                or take_by_coords(footprint, np.asarray(footprint.shape) // 2) != 1
            )
        ):
            return

        desired_out = sk_op(inp, footprint)

        assert_eq(
            imops_op(inp, footprint, backend=backend, boxed=boxed),
            desired_out,
            err_msg=f'{i, shape, footprint, box_coord if boxed else None}',
        )
