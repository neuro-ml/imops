from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose

from imops._configs import numeric_configs
from imops.backend import Backend
from imops.numeric import _STR_TYPES, pointwise_add


np.random.seed(1337)

assert_eq = np.testing.assert_array_equal
n_samples = 8


@dataclass
class Alien8(Backend):
    pass


@pytest.fixture(params=numeric_configs, ids=map(str, numeric_configs))
def backend(request):
    return request.param


@pytest.fixture(params=range(1, 8))
def num_threads(request):
    return request.param


@pytest.fixture(params=_STR_TYPES)
def dtype(request):
    return request.param


@pytest.fixture(params=['uint16', 'uint32', 'uint64', 'float128', 'complex64'])
def bad_dtype(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien8(), 'Alien9'], ids=['empty', 'Alien8', 'Alien9'])
def test_alien_backend(alien_backend):
    nums = np.random.randn(1337)

    with pytest.raises(ValueError):
        pointwise_add(nums, 1, backend=alien_backend)


def test_empty_add(backend, num_threads, dtype):
    nums1 = np.array([], dtype=dtype)
    nums2 = np.array([], dtype=dtype)

    out = pointwise_add(nums1, nums2, num_threads=num_threads, backend=backend)
    desired_out = nums1 + nums2

    assert_eq(out, desired_out)


def test_different_dtypes(backend):
    for dtype1, dtype2 in combinations(_STR_TYPES, 2):
        for _ in range(n_samples):
            shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

            nums1 = (32 * np.random.randn(*shape)).astype(dtype1)
            nums2 = (32 * np.random.randn(*shape)).astype(dtype2)

            with pytest.raises(ValueError):
                pointwise_add(nums1, nums2, backend=backend)


def test_bad_dtypes(backend, bad_dtype):
    for dtype1, dtype2 in combinations(_STR_TYPES, 2):
        for _ in range(n_samples):
            shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

            nums1 = (32 * np.random.randn(*shape)).astype(dtype1)
            nums2 = (32 * np.random.randn(*shape)).astype(dtype2)

            with pytest.raises(ValueError):
                pointwise_add(nums1, nums2, backend=backend)


def test_stress_pointwise_add(backend, num_threads, dtype):
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums1 = (32 * np.random.randn(*shape)).astype(dtype)
        nums2 = (
            (32 * np.random.randn(*shape)).astype(dtype)
            if np.random.binomial(1, 0.5)
            else np.dtype(dtype).type(32 * np.random.randn(1)[0])
        )

        out = pointwise_add(nums1, nums2, num_threads=num_threads, backend=backend)
        desired_out = nums1 + nums2

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(out, desired_out)
        else:
            allclose(out, desired_out)


def test_stress_pointwise_add_output(backend, num_threads, dtype):
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums1 = (32 * np.random.randn(*shape)).astype(dtype)
        old_nums1 = np.copy(nums1)
        output = np.empty_like(nums1)
        nums2 = (
            (32 * np.random.randn(*shape)).astype(dtype)
            if np.random.binomial(1, 0.5)
            else np.dtype(dtype).type(32 * np.random.randn(1)[0])
        )

        out = pointwise_add(nums1, nums2, output=output, num_threads=num_threads, backend=backend)
        desired_out = nums1 + nums2

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(out, desired_out)
            assert_eq(output, desired_out)
            assert_eq(nums1, old_nums1)
        else:
            allclose(out, desired_out)
            allclose(output, desired_out)
            allclose(nums1, old_nums1)


def test_stress_pointwise_add_inplace(backend, num_threads, dtype):
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums1 = (32 * np.random.randn(*shape)).astype(dtype)
        nums2 = (
            (32 * np.random.randn(*shape)).astype(dtype)
            if np.random.binomial(1, 0.5)
            else np.dtype(dtype).type(32 * np.random.randn(1)[0])
        )

        desired_out = nums1 + nums2
        out = pointwise_add(nums1, nums2, output=nums1, num_threads=num_threads, backend=backend)

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(out, desired_out)
            assert_eq(nums1, desired_out)
        else:
            allclose(out, desired_out)
            allclose(nums1, desired_out)
