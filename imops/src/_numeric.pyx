# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport numpy as np
from libc.stdint cimport uint16_t

from cython.parallel import prange


# https://stackoverflow.com/questions/47421443/using-half-precision-numpy-floats-in-cython
cdef extern from "numpy/halffloat.h":
    ctypedef uint16_t npy_half

    float npy_half_to_float(npy_half h) nogil
    npy_half npy_float_to_half(float f) nogil


ctypedef fused NUM:
    short
    int
    long long
    float
    double


ctypedef fused NUM_AND_NPY_HALF:
    NUM
    npy_half


# TODO: Generalize code below to n-d
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


def _pointwise_add_array_3d_fp16(
    npy_half[:, :, :] nums1,
    npy_half[:, :, :] nums2,
    npy_half[:, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t rows = out.shape[0], cols = out.shape[1], dims = out.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                out[i, j, k] = (npy_float_to_half(npy_half_to_float(nums1[i, j, k]) +
                                npy_half_to_float(nums2[i, j, k])))

    return np.asarray(out)


def _pointwise_add_array_4d_fp16(
    npy_half[:, :, :, :] nums1,
    npy_half[:, :, :, :] nums2,
    npy_half[:, :, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t dim1 = out.shape[0], dim2 = out.shape[1], dim3 = out.shape[2], dim4 = out.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    out[i1, i2, i3, i4] = (npy_float_to_half(npy_half_to_float(nums1[i1, i2, i3, i4]) +
                                           npy_half_to_float(nums2[i1, i2, i3, i4])))

    return np.asarray(out)


def _pointwise_add_value_3d_fp16(
    npy_half[:, :, :] nums,
    npy_half value,
    npy_half[:, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t rows = out.shape[0], cols = out.shape[1], dims = out.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                out[i, j, k] = npy_float_to_half(npy_half_to_float(nums[i, j, k]) + npy_half_to_float(value))

    return np.asarray(out)


def _pointwise_add_value_4d_fp16(
    npy_half[:, :, :, :] nums,
    npy_half value,
    npy_half[:, :, :, :] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    cdef Py_ssize_t dim1 = out.shape[0], dim2 = out.shape[1], dim3 = out.shape[2], dim4 = out.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    out[i1, i2, i3, i4] = (npy_float_to_half(npy_half_to_float(nums[i1, i2, i3, i4]) +
                                           npy_half_to_float(value)))

    return np.asarray(out)


def _fill_3d(NUM_AND_NPY_HALF[:, :, :] nums, NUM_AND_NPY_HALF value, Py_ssize_t num_threads) -> None:
    cdef Py_ssize_t rows = nums.shape[0], cols = nums.shape[1], dims = nums.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                nums[i, j, k] = value


def _fill_4d(NUM_AND_NPY_HALF[:, :, :, :] nums, NUM_AND_NPY_HALF value, Py_ssize_t num_threads) -> None:
    cdef Py_ssize_t dim1 = nums.shape[0], dim2 = nums.shape[1], dim3 = nums.shape[2], dim4 = nums.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    nums[i1, i2, i3, i4] = value


# FIXME: somehow `const NUM_AND_NPY_HALF` is not working
cpdef void _copy_3d(const NUM[:, :, :] nums1, NUM[:, :, :] nums2, Py_ssize_t num_threads):
    cdef Py_ssize_t rows = nums1.shape[0], cols = nums1.shape[1], dims = nums1.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                nums2[i, j, k] = nums1[i, j, k]


cpdef void _copy_4d(const NUM[:, :, :, :] nums1, NUM[:, :, :, :] nums2, Py_ssize_t num_threads):
    cdef Py_ssize_t dim1 = nums1.shape[0], dim2 = nums1.shape[1], dim3 = nums1.shape[2], dim4 = nums1.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    nums2[i1, i2, i3, i4] = nums1[i1, i2, i3, i4]


cpdef void _copy_3d_fp16(const npy_half[:, :, :] nums1, npy_half[:, :, :] nums2, Py_ssize_t num_threads):
    cdef Py_ssize_t rows = nums1.shape[0], cols = nums1.shape[1], dims = nums1.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                nums2[i, j, k] = nums1[i, j, k]


cpdef void _copy_4d_fp16(const npy_half[:, :, :, :] nums1, npy_half[:, :, :, :] nums2, Py_ssize_t num_threads):
    cdef Py_ssize_t dim1 = nums1.shape[0], dim2 = nums1.shape[1], dim3 = nums1.shape[2], dim4 = nums1.shape[3]
    cdef Py_ssize_t i1, i2, i3, i4

    for i1 in prange(dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(dim2):
            for i3 in prange(dim3):
                for i4 in prange(dim4):
                    nums2[i1, i2, i3, i4] = nums1[i1, i2, i3, i4]
