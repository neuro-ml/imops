from dataclasses import dataclass

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt
from skimage.morphology import (
    binary_closing as sk_binary_closing,
    binary_dilation as sk_binary_dilation,
    binary_erosion as sk_binary_erosion,
    binary_opening as sk_binary_opening,
)

from imops._configs import morphology_configs
from imops.backend import Backend, Scipy
from imops.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, distance_transform_edt
from imops.pad import restore_crop
from imops.utils import make_immutable


np.random.seed(1337)

assert_eq = np.testing.assert_array_equal
n_samples = 8
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

    with pytest.raises(ValueError):
        distance_transform_edt(inp, backend=alien_backend)


def test_single_threaded_warning(pair):
    _, imops_op = pair
    with pytest.warns(UserWarning):
        imops_op(np.ones(1), num_threads=2, backend='Scipy')
    with pytest.warns(UserWarning):
        distance_transform_edt(np.ones(1), num_threads=2, backend='Scipy')


def test_empty(pair, backend):
    _, imops_op = pair
    with pytest.raises(RuntimeError):
        imops_op(np.ones(1), np.array([]))


def test_wrong_footprint(pair, backend):
    imops_op = pair[1]

    inp = np.ones((3, 4, 5))
    footprint = np.ones((1, 2))

    if backend != Scipy():
        with pytest.raises(ValueError):
            imops_op(inp, footprint=footprint, backend=backend)


def test_scipy_warning(pair, backend):
    imops_op = pair[1]

    inp = np.ones((3, 4, 5, 6))

    if backend != Scipy():
        with pytest.warns(UserWarning):
            imops_op(inp, backend=backend)

        with pytest.warns(UserWarning):
            distance_transform_edt(inp, backend=backend)


def test_output_shape_mismatch(pair, backend, boxed):
    imops_op = pair[1]
    boxed = boxed and backend != Scipy()

    inp = np.ones((3, 4, 5))
    output = np.empty((3, 4, 6), dtype=bool)

    with pytest.raises(ValueError):
        imops_op(inp, output=output, backend=backend, boxed=boxed)


def test_output_c_contiguity_mismatch(pair, backend, boxed):
    imops_op = pair[1]
    boxed = boxed and backend != Scipy()

    inp = np.ones((3, 4, 5))
    output = np.empty((6, 8, 10), dtype=bool)[::2, ::2, ::2]

    with pytest.raises(ValueError):
        imops_op(inp, output=output, backend=backend, boxed=boxed)


def test_trivial_input_warning(pair, backend, boxed):
    imops_op = pair[1]

    inp = np.ones((3, 4, 5)).astype(bool)

    if backend != Scipy():
        with pytest.warns(UserWarning):
            imops_op(inp, backend=backend, boxed=boxed)

        with pytest.warns(UserWarning):
            imops_op((1 - inp).astype(bool), backend=backend, boxed=boxed)


def test_stress(pair, backend, footprint_shape_modifier, boxed):
    # FIXME
    def take_by_coords(array, coords):
        copy_array = np.copy(array)
        for coord in coords:
            copy_array = copy_array[coord]

        return copy_array

    sk_op, imops_op = pair

    for i in range(2 * n_samples):
        shape = np.random.randint(64, 128, size=np.random.randint(1, 4))

        if boxed:
            box_size = np.asarray([np.random.randint(s // 3, s + 1) for s in shape])
            box_pos = np.asarray([np.random.randint(0, s - bs + 1) for bs, s in zip(box_size, shape)])
            box_coord = np.array([box_pos, box_pos + box_size])
            inp = np.random.binomial(1, 0.7, box_size)
            inp = restore_crop(inp, box_coord, shape, 0).astype(bool)
        else:
            inp = np.random.binomial(1, 0.5, shape).astype(bool)

        if np.random.binomial(1, 0.5):
            make_immutable(inp)

        footprint_shape = footprint_shape_modifier(np.random.randint(1, 4, size=inp.ndim))
        footprint = np.random.binomial(1, 0.5, footprint_shape) if np.random.binomial(1, 0.5) else None

        if footprint is not None and np.random.binomial(1, 0.5):
            make_immutable(footprint)

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
        output = np.empty_like(inp)

        if np.random.binomial(1, 0.5) or boxed:
            output = imops_op(inp, footprint, backend=backend, boxed=boxed)
        else:
            imops_op(inp, footprint, output=output, backend=backend, boxed=boxed)

        assert_eq(
            output,
            desired_out,
            err_msg=f'{i, shape, footprint, box_coord if boxed else None}',
        )


@pytest.fixture(params=range(1, 4))
def num_threads(request):
    return request.param


@pytest.fixture(params=[False, True])
def return_distances(request):
    return request.param


@pytest.fixture(params=[False, True])
def return_indices(request):
    return request.param


@pytest.fixture(params=[False, True])
def sampling_enabled(request):
    return request.param


def test_stress_edt(backend, num_threads, return_distances, return_indices, sampling_enabled):
    if not return_distances and not return_indices:
        with pytest.raises(RuntimeError):
            distance_transform_edt(
                np.ones(1),
                return_distances=return_distances,
                return_indices=return_indices,
                num_threads=num_threads,
                backend=backend,
            )
        return

    for _ in range(n_samples):
        sampling = None
        size = np.random.randint(1, 4)
        if sampling_enabled:
            sampling = tuple(np.random.uniform(0.5, 1.5, size=size))
        shape = np.random.randint(64, 128, size=size)
        x = np.random.randint(5, size=shape)

        if np.random.binomial(1, 0.5):
            make_immutable(x)

        out = distance_transform_edt(
            x,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
            num_threads=num_threads,
            backend=backend,
        )
        ref_out = scipy_distance_transform_edt(
            x, sampling=sampling, return_distances=return_distances, return_indices=return_indices
        )

        if isinstance(out, tuple):
            np.testing.assert_allclose(out[0], ref_out[0], rtol=1e-6)
            np.testing.assert_equal(out[1], ref_out[1])
        else:
            np.testing.assert_allclose(out, ref_out, rtol=1e-6)
