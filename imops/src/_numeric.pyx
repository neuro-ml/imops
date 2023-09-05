# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport numpy as np

from cython.parallel import prange


ctypedef fused NUM:
    short
    int
    long long
    float
    double


def _sum_1d(NUM[:] nums, Py_ssize_t num_threads) -> NUM:
    cdef NUM res = 0
    cdef Py_ssize_t i, len_nums = len(nums)

    for i in prange(len_nums, num_threads=num_threads, nogil=True):
        res += nums[i]

    return res


# TODO: Generalize code below to n-d
def _pointwise_mul_3d(
    NUM[:, :, :] nums1,
    NUM[:, :, :] nums2,
    Py_ssize_t[:] res_shape,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef NUM[:, :, ::1] contiguous_nums1 = np.ascontiguousarray(nums1), contiguous_nums2 = np.ascontiguousarray(nums2)
    cdef Py_ssize_t rows = res_shape[0], cols = res_shape[1], dims = res_shape[2]

    cdef char[:] broadcast_mask1 = np.array([x == y for x, y in zip(res_shape, nums1.shape)], dtype=np.int8)
    cdef char[:] broadcast_mask2 = np.array([x == y for x, y in zip(res_shape, nums2.shape)], dtype=np.int8)

    cdef NUM[:, :, ::1] mul = np.empty_like(nums1, shape=res_shape)
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                mul[i, j, k] = (
                    contiguous_nums1[
                        i * broadcast_mask1[0],
                        j * broadcast_mask1[1],
                        k * broadcast_mask1[2]
                    ] *
                    contiguous_nums2[
                        i * broadcast_mask2[0],
                        j * broadcast_mask2[1],
                        k * broadcast_mask2[2]
                    ]
                )

    return np.asarray(mul)


def _pointwise_add_array_3d(
    NUM[:, :, :] nums1,
    NUM[:, :, :] nums2,
    NUM[:, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t rows = out.shape[0], cols = out.shape[1], dims = out.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                out[i, j, k] = nums1[i, j, k] + nums2[i, j, k]

    return np.asarray(out)


def _pointwise_add_array_4d(
    NUM[:, :, :, :] nums1,
    NUM[:, :, :, :] nums2,
    NUM[:, :, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t dim1 = out.shape[0], dim2 = out.shape[1], dim3 = out.shape[2], dim4 = out.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    out[i1, i2, i3, i4] = nums1[i1, i2, i3, i4] + nums2[i1, i2, i3, i4]

    return np.asarray(out)


def _pointwise_add_value_3d(
    NUM[:, :, :] nums,
    NUM value,
    NUM[:, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t rows = out.shape[0], cols = out.shape[1], dims = out.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                out[i, j, k] = nums[i, j, k] + value

    return np.asarray(out)


def _pointwise_add_value_4d(
    NUM[:, :, :, :] nums,
    NUM value,
    NUM[:, :, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t dim1 = out.shape[0], dim2 = out.shape[1], dim3 = out.shape[2], dim4 = out.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    out[i1, i2, i3, i4] = nums[i1, i2, i3, i4] + value

    return np.asarray(out)
