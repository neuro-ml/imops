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


def _parallel_sum(NUM[:] nums, Py_ssize_t num_threads) -> NUM:
    cdef NUM res = 0
    cdef Py_ssize_t i, len_nums = len(nums)

    for i in prange(len_nums, num_threads=num_threads, nogil=True):
        res += nums[i]

    return res


def _parallel_pointwise_mul(
    NUM[:, :, :] nums1,
    NUM[:, :, :] nums2,
    Py_ssize_t[:] res_shape,
    Py_ssize_t num_threads
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
