import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.measure import label as sk_label

from imops.measure import label


@pytest.fixture(params=[1, 2, 3, 4])
def connectivity(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def ndim(request):
    return request.param


def test_dtype(connectivity, ndim):
    connectivity = min(connectivity, ndim)

    for dtype in (
        bool,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ):
        inp_dtype = np.random.randint(0, 5, size=np.random.randint(32, 64, size=ndim),).astype(
            dtype
        )[0 if ndim == 4 and dtype != bool else ...]
        connectivity = min(connectivity, inp_dtype.ndim)

        assert_array_equal(
            sk_label(inp_dtype, connectivity=connectivity),
            label(inp_dtype, connectivity=connectivity),
            err_msg=str(dtype),
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
        assert_array_equal(
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

    assert_array_equal(
        sk_label(inp, connectivity=connectivity),
        label(inp, connectivity=connectivity),
        err_msg=f'{connectivity, ndim}',
    )


def test_zeros(connectivity, ndim):
    connectivity = min(connectivity, ndim)
    inp = np.zeros(np.random.randint(32, 64, size=ndim))
    if ndim == 4:
        inp = inp.astype(bool)

    assert_array_equal(
        sk_label(inp, connectivity=connectivity),
        label(inp, connectivity=connectivity),
        err_msg=f'{connectivity, ndim}',
    )


def test_stress(connectivity, ndim):
    connectivity = min(connectivity, ndim)

    for _ in range(32):
        inp = (
            np.random.binomial(1, 0.5, size=np.random.randint(32, 64, size=ndim)) > 0
            if ndim == 4 or np.random.binomial(1, 0.2)
            else np.random.randint(0, 5, size=np.random.randint(32, 64, size=ndim))
        )
        sk_labeled, sk_num_components = sk_label(inp, connectivity=connectivity, return_num=True)
        labeled, num_components = label(inp, connectivity=connectivity, return_num=True)

        assert_array_equal(
            sk_labeled,
            labeled,
            err_msg=f'{connectivity, ndim, inp.shape}',
        )
        assert sk_num_components == num_components, f'{connectivity, ndim, inp.shape}'