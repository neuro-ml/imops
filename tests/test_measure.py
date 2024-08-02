from dataclasses import dataclass
from platform import python_version

import numpy as np
import pytest
from fastremap import unique
from numpy.testing import assert_allclose as allclose
from scipy.ndimage import center_of_mass as scipy_center_of_mass
from skimage.measure import label as sk_label

from imops._configs import measure_configs
from imops.backend import Backend, Scipy
from imops.measure import center_of_mass, label
from imops.utils import make_immutable


np.random.seed(1337)

assert_eq = np.testing.assert_array_equal
n_samples = 8


@dataclass
class Alien10(Backend):
    pass


@pytest.fixture(params=[1, 2, 3, 4])
def connectivity(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def ndim(request):
    return min(3, request.param) if python_version()[:3] == '3.6' else request.param


@pytest.fixture(
    params=['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
)
def mask_dtype(request):
    return request.param


def test_dtype(connectivity, ndim, mask_dtype):
    connectivity = min(connectivity, ndim)

    inp = np.random.randint(0, 5, size=np.random.randint(32, 64, size=ndim)).astype(mask_dtype)[
        0 if ndim == 4 and dtype != bool else ...
    ]
    connectivity = min(connectivity, inp.ndim)

    assert_eq(
        sk_label(inp, connectivity=connectivity),
        label(inp, connectivity=connectivity),
        err_msg=str(mask_dtype),
    )


def test_background(connectivity, ndim):
    connectivity = min(connectivity, ndim)
    booled = ndim == 4

    inp = (
        np.random.randint(
            0,
            5,
            size=np.random.randint(32, 64, size=ndim),
        )
        if not booled
        else np.random.binomial(1, 0.5, size=np.random.randint(32, 64, size=ndim)) > 0
    )

    for background in [0, 1, 2, 3, 4]:
        background = background > 0 if booled else background
        assert_eq(
            sk_label(inp, connectivity=connectivity, background=background),
            label(inp, connectivity=connectivity, background=background),
            err_msg=f'{connectivity, ndim, background}',
        )


def test_connectivity_exception(connectivity, ndim):
    if connectivity > ndim:
        with pytest.raises(ValueError):
            label(np.random.randint(0, 5, size=np.random.randint(32, 64, size=ndim)), connectivity=connectivity)


def test_ones(connectivity, ndim):
    connectivity = min(connectivity, ndim)
    inp = np.ones(np.random.randint(32, 64, size=ndim))
    if ndim == 4:
        inp = inp.astype(bool)

    assert_eq(
        sk_label(inp, connectivity=connectivity),
        label(inp, connectivity=connectivity),
        err_msg=f'{connectivity, ndim}',
    )


def test_zeros(connectivity, ndim):
    connectivity = min(connectivity, ndim)
    inp = np.zeros(np.random.randint(32, 64, size=ndim))
    if ndim == 4:
        inp = inp.astype(bool)

    assert_eq(
        sk_label(inp, connectivity=connectivity),
        label(inp, connectivity=connectivity),
        err_msg=f'{connectivity, ndim}',
    )


def test_out_dtype(connectivity, ndim):
    connectivity = min(connectivity, ndim)

    for dtype in [np.uint16, np.uint32, np.uint64]:

        inp = np.zeros(np.random.randint(32, 64, size=ndim))
        if ndim == 4:
            inp = inp.astype(bool)

        out = label(inp, connectivity=connectivity, dtype=dtype)
        assert_eq(
            sk_label(inp, connectivity=connectivity).astype(dtype),
            out,
            err_msg=f'{connectivity, ndim, dtype}',
        )


def test_multiple_output():
    inp = np.array(
        [
            [1, 0, 0, 1],
            [2, 2, 1, 1],
            [2, 2, 2, 1],
            [1, 1, 0, 0],
        ]
    )

    labeled, num_components = label(inp, return_num=True)
    assert num_components == 4, labeled

    labeled, num_components, labels = label(inp, return_num=True, return_labels=True)
    assert num_components == 4
    assert (labels == np.array([1, 2, 3, 4])).all()

    labeled, num_components, labels, sizes = label(inp, return_num=True, return_labels=True, return_sizes=True)
    assert num_components == 4
    assert (labels == np.array([1, 2, 3, 4])).all()
    assert (sorted(sizes) == np.array([1, 2, 4, 5])).all()

    labeled, num_components, sizes = label(inp, return_num=True, return_sizes=True)
    assert num_components == 4
    assert (sorted(sizes) == np.array([1, 2, 4, 5])).all()

    labeled, sizes = label(inp, return_sizes=True)
    assert (sorted(sizes) == np.array([1, 2, 4, 5])).all()

    labeled, labels = label(inp, return_labels=True)
    assert (labels == np.array([1, 2, 3, 4])).all()


def test_no_background():
    inp = np.ones((3, 3))

    labeled, num_components, labels, sizes = label(inp, return_num=True, return_labels=True, return_sizes=True)

    assert num_components == 1
    assert len(labels) == 1
    assert (labels == np.array([1])).all()
    assert (sorted(sizes) == np.array([9])).all()
    assert len(sizes) == 1


def test_no_background2():
    inp = np.ones((3, 3))
    inp[0, 0] = 2

    labeled, num_components, labels, sizes = label(inp, return_num=True, return_labels=True, return_sizes=True)

    assert num_components == 2
    assert len(labels) == 2
    assert (labels == np.array([1, 2])).all()
    assert len(sizes) == 2
    assert (sorted(sizes) == np.array([1, 8])).all()


def test_stress(connectivity, ndim):
    connectivity = min(connectivity, ndim)

    for _ in range(2 * n_samples):
        inp = (
            np.random.binomial(1, 0.5, size=np.random.randint(32, 64, size=ndim)) > 0
            if ndim == 4 or np.random.binomial(1, 0.2)
            else np.random.randint(0, 5, size=np.random.randint(32, 64, size=ndim))
        )

        sk_labeled, sk_num_components = sk_label(inp, connectivity=connectivity, return_num=True)
        labeled, num_components = label(inp, connectivity=connectivity, return_num=True)

        assert_eq(
            sk_labeled,
            labeled,
            err_msg=f'{connectivity, ndim, inp.shape}',
        )
        assert sk_num_components == num_components, f'{connectivity, ndim, inp.shape}'


@pytest.fixture(
    params=['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
)
def dtype(request):
    return request.param


@pytest.fixture(params=['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'])
def label_dtype(request):
    return request.param


@pytest.fixture(params=range(1, 4))
def num_threads(request):
    return request.param


@pytest.fixture(params=measure_configs, ids=map(str, measure_configs))
def backend(request):
    return request.param


def test_both_specified(backend):
    inp = np.random.randn(32, 32)
    labels = np.random.randint(0, 16, size=inp.shape)
    index = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        center_of_mass(inp, labels=labels, backend=backend)

    with pytest.raises(ValueError):
        center_of_mass(inp, index=index, backend=backend)


def test_noncontiguous_index(backend, label_dtype):
    for _ in range(n_samples):
        inp = np.random.randn(32, 32) + 4
        labels = np.random.randint(0, 16, size=inp.shape).astype(label_dtype)
        index = np.random.permutation(np.arange(8)).astype(label_dtype) if label_dtype != 'bool' else [True, False]

        out = center_of_mass(inp, labels, index, backend=backend)
        desired_out = scipy_center_of_mass(inp, labels, index)

        allclose(out, desired_out, rtol=1e-6)


@pytest.mark.parametrize('alien_backend', ['', Alien10(), 'Alien11'], ids=['empty', 'Alien10', 'Alien11'])
def test_alien_backend(alien_backend):
    inp = np.random.randn(32, 32)
    labels = np.random.randint(0, 16, size=inp.shape)
    index = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        center_of_mass(inp, backend=alien_backend)

    with pytest.raises(ValueError):
        center_of_mass(inp, labels=labels, index=index, backend=alien_backend)


def test_center_of_mass(num_threads, backend, dtype):
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 4))
        inp = (
            np.random.binomial(1, 0.5, shape).astype(dtype)
            if dtype == 'bool' or 'u' in dtype
            else (32 * np.random.randn(*shape)).astype(dtype) + 4
        )
        if np.random.binomial(1, 0.5):
            make_immutable(inp)

        out = center_of_mass(inp, num_threads=num_threads, backend=backend)
        desired_out = scipy_center_of_mass(inp)

        assert isinstance(out, tuple)
        assert isinstance(desired_out, tuple)
        allclose(out, desired_out, err_msg=(inp, inp.shape), rtol=1e-3)


def test_scipy_warning(num_threads, backend, dtype):
    inp = np.random.randn(32, 32, 32, 32).astype(dtype)

    with pytest.warns(UserWarning):
        center_of_mass(inp, num_threads=num_threads, backend=backend)


def test_labels_index_dtype(backend):
    inp = np.random.randn(32, 32, 32)
    labels = np.random.randint(0, 4, size=inp.shape)
    index = np.array([False, True])

    if backend != Scipy():
        with pytest.raises(ValueError):
            center_of_mass(inp, labels=labels, index=index, backend=backend)


def test_labels_shape_mismatch(backend):
    inp = np.random.randn(32, 32, 32)
    labels = np.random.randint(0, 4, size=(1, 2, 3))
    index = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        center_of_mass(inp, labels=labels, index=index, backend=backend)

    labels = np.random.randint(0, 4, size=(1, 2))

    with pytest.raises(ValueError):
        center_of_mass(inp, labels=labels, index=index, backend=backend)


def test_not_unique_index(backend):
    inp = np.random.randn(32, 32, 32)
    labels = np.random.randint(0, 4, size=(32, 32, 32))
    index = np.array([1, 1, 2])

    if backend != Scipy():
        with pytest.raises(ValueError):
            center_of_mass(inp, labels=labels, index=index, backend=backend)


def test_labeled_center_of_mass(backend, dtype, label_dtype):
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 4))
        inp = (
            np.random.binomial(1, 0.5, shape).astype(dtype)
            if dtype == 'bool'
            else (32 * np.random.randn(*shape)).astype(dtype) + 4
        )
        labels = np.random.randint(0, 2 if label_dtype == 'bool' else 16, size=shape, dtype=label_dtype)
        index = (
            unique(np.random.randint(0, 32, size=16)).astype(label_dtype)
            if label_dtype != 'bool'
            else np.random.choice(np.array([[False], [True], [False, True], [True, False]], dtype=object))
        )
        if np.random.binomial(1, 0.5):
            make_immutable(inp)
        if np.random.binomial(1, 0.5):
            make_immutable(labels)

        out = center_of_mass(inp, labels, index, num_threads=1, backend=backend)
        desired_out = scipy_center_of_mass(inp, labels, index)

        for x, y in zip(out, desired_out):
            assert isinstance(x, tuple)
            assert isinstance(y, tuple)

        allclose(out, desired_out, err_msg=(inp, inp.shape), rtol=1e-3)
