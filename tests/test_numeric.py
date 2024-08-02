from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose as allclose

from imops._configs import numeric_configs
from imops.backend import Backend
from imops.numeric import _STR_TYPES, copy, fill_, full, pointwise_add
from imops.utils import make_immutable


np.random.seed(1337)

assert_eq = np.testing.assert_array_equal
n_samples = 8


@dataclass
class Alien8(Backend):
    pass


@pytest.fixture(params=numeric_configs, ids=map(str, numeric_configs))
def backend(request):
    return request.param


@pytest.fixture(params=range(1, 9))
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

    with pytest.raises(ValueError):
        fill_(nums, 42, backend=alien_backend)

    with pytest.raises(ValueError):
        copy(nums, backend=alien_backend)


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
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums = (32 * np.random.randn(*shape)).astype(bad_dtype)

        with pytest.raises(ValueError):
            pointwise_add(nums, 42, backend=backend)


def test_bad_output_shape(backend, num_threads, dtype):
    shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
    output_shape = np.random.randint(32, 64, size=len(shape))

    while not (shape != output_shape).any():
        output_shape = np.random.randint(32, 64, size=len(shape))

    nums = (32 * np.random.randn(*shape)).astype(dtype)
    output = np.empty(output_shape, dtype=dtype)

    with pytest.raises(ValueError):
        pointwise_add(nums, 42, output=output, num_threads=num_threads, backend=backend)


def test_bad_output_dtype(backend, num_threads, dtype):
    for output_dtype in _STR_TYPES:
        if output_dtype == dtype:
            continue

        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums = (32 * np.random.randn(*shape)).astype(dtype)
        output = np.empty_like(nums, dtype=output_dtype)

        with pytest.raises(ValueError):
            pointwise_add(nums, 42, output=output, num_threads=num_threads, backend=backend)


def test_bad_scalar_summand_dtype(backend, num_threads, dtype, bad_dtype):
    shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
    nums = (32 * np.random.randn(*shape)).astype(dtype)

    with pytest.raises(ValueError):
        pointwise_add(nums, np.dtype(bad_dtype).type(42), num_threads=num_threads, backend=backend)


def test_copy_bad_output_dtype(backend, num_threads, dtype):
    for output_dtype in _STR_TYPES:
        if output_dtype == dtype:
            continue

        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums = (32 * np.random.randn(*shape)).astype(dtype)
        output = np.empty_like(nums, dtype=output_dtype)

        with pytest.raises(ValueError):
            copy(nums, output=output, num_threads=num_threads, backend=backend)


def test_copy_bad_output_shape(backend, num_threads, dtype):
    shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
    output_shape = np.random.randint(32, 64, size=len(shape))

    while not (shape != output_shape).any():
        output_shape = np.random.randint(32, 64, size=len(shape))

    nums = (32 * np.random.randn(*shape)).astype(dtype)
    output = np.empty(output_shape, dtype=dtype)

    with pytest.raises(ValueError):
        copy(nums, output=output, num_threads=num_threads, backend=backend)


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


def test_stress_fill_(backend, num_threads, dtype):
    def sample_value(dtype):
        x = 32 * np.random.randn(1)
        if isinstance(x, np.ndarray):
            x = x[0]
        x = dtype.type(x)

        dice = np.random.randint(3)

        if dice == 0:
            return x

        if dice == 1:
            return int(x)

        return float(x)

    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums = (32 * np.random.randn(*shape)).astype(dtype)
        nums_copy = np.copy(nums)

        value = sample_value(nums.dtype)

        fill_(nums, value, num_threads, backend)
        nums_copy.fill(value)

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(nums, nums_copy)
        else:
            allclose(nums, nums_copy)


def test_stress_full(backend, num_threads, dtype):
    def sample_value(dtype):
        x = 32 * np.random.randn(1)
        if isinstance(x, np.ndarray):
            x = x[0]
        x = dtype.type(x)

        dice = np.random.randint(3)

        if dice == 0:
            return x

        if dice == 1:
            return int(x)

        return float(x)

    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))
        fill_value = sample_value(np.zeros(1, dtype=dtype).dtype)

        dtype_or_none = dtype if np.random.binomial(1, 0.5) else None

        nums = full(shape, fill_value, dtype_or_none, num_threads=num_threads, backend=backend)
        desired_nums = np.full(shape, fill_value, dtype_or_none)

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(nums, desired_nums)
        else:
            allclose(nums, desired_nums)


def test_stress_copy(backend, num_threads, dtype):
    for _ in range(n_samples):
        shape = np.random.randint(32, 64, size=np.random.randint(1, 5))

        nums = (32 * np.random.randn(*shape)).astype(dtype)
        if np.random.binomial(1, 0.5):
            make_immutable(nums)

        old_nums = np.copy(nums)
        copy_nums = copy(
            nums,
            output=None if np.random.binomial(1, 0.5) else np.empty_like(nums),
            num_threads=num_threads,
            backend=backend,
        )

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(old_nums, copy_nums)
            assert_eq(nums, old_nums)
        else:
            allclose(old_nums, copy_nums)
            allclose(nums, old_nums)

        copy_nums[0] = 0

        if dtype in ('int16', 'int32', 'int64'):
            assert_eq(nums, old_nums)
        else:
            allclose(nums, old_nums)
