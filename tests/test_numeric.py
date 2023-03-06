from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose

from imops._configs import numeric_configs
from imops.backend import Backend
from imops.numeric import parallel_pointwise_mul, parallel_sum


assert_eq = np.testing.assert_array_equal


@dataclass
class Alien8(Backend):
    pass


@pytest.fixture(params=numeric_configs, ids=map(str, numeric_configs))
def backend(request):
    return request.param


@pytest.fixture(params=range(1, 9))
def num_threads(request):
    return request.param


@pytest.fixture(params=['int16', 'int32', 'int64', 'float32', 'float64'])
def dtype(request):
    return request.param


@pytest.mark.parametrize('alien_backend', ['', Alien8(), 'Alien9'], ids=['empty', 'Alien8', 'Alien9'])
def test_alien_backend(alien_backend):
    nums1 = np.random.randn(1337)
    nums2 = np.random.randn(1337)

    with pytest.raises(ValueError):
        parallel_sum(nums1, backend=alien_backend)

    with pytest.raises(ValueError):
        parallel_pointwise_mul(nums1, nums2, backend=alien_backend)


def test_parallel_sum_ndim_mismatch(backend, num_threads, dtype):
    with pytest.raises(ValueError):
        parallel_sum(np.ones((2, 2), dtype=dtype), num_threads=num_threads, backend=backend)


def test_pointwise_mul_size_mismatch(backend, num_threads, dtype):
    with pytest.raises(ValueError):
        parallel_pointwise_mul(
            np.ones(4, dtype=dtype), np.ones(2, dtype=dtype), num_threads=num_threads, backend=backend
        )
    with pytest.raises(ValueError):
        parallel_pointwise_mul(
            np.ones(3, dtype=dtype), np.ones(2, dtype=dtype), num_threads=num_threads, backend=backend
        )
    with pytest.raises(ValueError):
        parallel_pointwise_mul(np.array([]), np.ones(2, dtype=dtype), num_threads=num_threads, backend=backend)

    with pytest.raises(ValueError):
        parallel_pointwise_mul(np.ones((1, 2, 3)), np.ones((1, 2, 3, 4)), num_threads=num_threads, backend=backend)


def test_empty_sum(backend, num_threads, dtype):
    nums = np.array([], dtype=dtype)

    out = parallel_sum(nums, num_threads=num_threads, backend=backend)
    desired_out = np.sum(nums)

    assert_eq(out, desired_out)
    assert out.dtype == desired_out.dtype


def test_empty_pointwise_mul(backend, num_threads, dtype):
    nums1 = np.array([], dtype=dtype)
    nums2 = np.array([], dtype=dtype)

    out = parallel_pointwise_mul(nums1, nums2, num_threads=num_threads, backend=backend)
    desired_out = nums1 * nums2

    assert_eq(out, desired_out)
    assert out.dtype == desired_out.dtype


def test_stress_sum(backend, num_threads, dtype):
    for _ in range(32):
        nums = (32 * np.random.randn(np.random.randint(1, 10**4))).astype(dtype)

        out = parallel_sum(nums, num_threads=num_threads, backend=backend)
        desired_out = np.sum(nums)

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(out, desired_out)
        else:
            allclose(out, desired_out, rtol=1e-4 if dtype == 'float32' else 1e-7)


def test_stress_pointwise_mul(backend, num_threads, dtype):
    for _ in range(32):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 4))

        nums1 = (32 * np.random.randn(*shape)).astype(dtype)
        nums2 = (32 * np.random.randn(*shape)).astype(dtype)

        out = parallel_pointwise_mul(nums1, nums2, num_threads=num_threads, backend=backend)
        desired_out = nums1 * nums2

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(out, desired_out)
        else:
            allclose(out, desired_out, rtol=1e-4 if dtype == 'float32' else 1e-7)


def test_broadcast_pointwise_mul(backend, num_threads, dtype):
    for _ in range(32):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 4))

        nums1 = (32 * np.random.randn(*[x if np.random.binomial(1, 0.7) else 1 for x in shape])).astype(dtype)
        nums2 = (32 * np.random.randn(*[x if np.random.binomial(1, 0.7) else 1 for x in shape])).astype(dtype)

        out = parallel_pointwise_mul(nums1, nums2, num_threads=num_threads, backend=backend)
        desired_out = nums1 * nums2

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(out, desired_out)
        else:
            allclose(out, desired_out, rtol=1e-4 if dtype == 'float32' else 1e-7)
