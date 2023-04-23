# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport cython
cimport numpy as np


ctypedef cython.integral INT


cdef inline Py_ssize_t _find(INT num, INT[:] nums) nogil:
    cdef Py_ssize_t i

    for i in range(len(nums)):
        if nums[i] == num:
            return i

    return -1


def _labeled_center_of_mass(double[:, :, :] nums, INT[:, :, :] labels, INT[:] index) -> np.ndarray:
    cdef double[:, :, ::1] contiguous_nums = np.ascontiguousarray(nums)
    cdef INT[:, :, ::1] contiguous_labels = np.ascontiguousarray(labels)

    cdef Py_ssize_t index_len = len(index)

    cdef double[:, ::1] output = np.zeros((index_len, 3))
    cdef double[:] normalizers = np.zeros(index_len)

    cdef Py_ssize_t rows = nums.shape[0], cols = nums.shape[1], dims = nums.shape[2]
    cdef Py_ssize_t i, j, k, pos

    # FIXME: why prange here makes everything significantly slower?
    for i in range(rows):
        for j in range(cols):
            for k in range(dims):
                pos = _find(contiguous_labels[i, j, k], index)

                if pos != -1:
                    normalizers[pos] += contiguous_nums[i, j, k]

                    output[pos, 0] += contiguous_nums[i, j, k] * i
                    output[pos, 1] += contiguous_nums[i, j, k] * j
                    output[pos, 2] += contiguous_nums[i, j, k] * k

    for i in range(index_len):
        for j in range(3):
            output[i, j] /= normalizers[i]

    return np.asarray(output)


def _center_of_mass(double[:, :, :] nums) -> np.ndarray:
    cdef double[:, :, ::1] contiguous_nums = np.ascontiguousarray(nums)

    cdef double[:] output = np.zeros(3)
    cdef double normalizer = 0

    cdef Py_ssize_t rows = nums.shape[0], cols = nums.shape[1], dims = nums.shape[2]
    cdef Py_ssize_t i, j, k

    # TODO: Use prange but consider critical section
    for i in range(rows):
        for j in range(cols):
            for k in range(dims):
                normalizer += contiguous_nums[i, j, k]

                output[0] += contiguous_nums[i, j, k] * i
                output[1] += contiguous_nums[i, j, k] * j
                output[2] += contiguous_nums[i, j, k] * k

    for i in range(3):
        output[i] /= normalizer

    return np.asarray(output)
