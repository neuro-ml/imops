# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange


ctypedef cython.floating FLOAT

ctypedef fused LABEL:
    signed char
    short
    int
    long long
    unsigned char
    unsigned short
    unsigned int
    unsigned long long


cdef inline Py_ssize_t _find(LABEL num, const LABEL[:] nums) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(len(nums)):
        if nums[i] == num:
            return i

    return -1


def _labeled_center_of_mass(const FLOAT[:, :, :] nums, const LABEL[:, :, :] labels,
                            const LABEL[:] index) -> np.ndarray:
    cdef const FLOAT[:, :, ::1] contiguous_nums = np.ascontiguousarray(nums)
    cdef const LABEL[:, :, ::1] contiguous_labels = np.ascontiguousarray(labels)
    cdef const LABEL[:] contiguous_index = np.ascontiguousarray(index)

    cdef Py_ssize_t index_len = len(index)

    cdef FLOAT[:, ::1] output = np.zeros_like(nums, shape=(index_len, 3))
    cdef FLOAT[:] normalizers = np.zeros_like(nums, shape=(index_len,))

    cdef Py_ssize_t rows = nums.shape[0], cols = nums.shape[1], dims = nums.shape[2]
    cdef Py_ssize_t i, j, k, pos

    # FIXME: why prange here makes everything significantly slower?
    for i in range(rows):
        for j in range(cols):
            for k in range(dims):
                pos = _find(contiguous_labels[i, j, k], contiguous_index)

                if pos != -1:
                    normalizers[pos] += contiguous_nums[i, j, k]

                    output[pos, 0] += contiguous_nums[i, j, k] * i
                    output[pos, 1] += contiguous_nums[i, j, k] * j
                    output[pos, 2] += contiguous_nums[i, j, k] * k

    for i in range(index_len):
        for j in range(3):
            output[i, j] /= normalizers[i]

    return np.asarray(output)


def _center_of_mass(const FLOAT[:, :, :] nums, Py_ssize_t num_threads) -> np.ndarray:
    cdef const FLOAT[:, :, ::1] contiguous_nums = np.ascontiguousarray(nums)

    cdef FLOAT output_x = 0, output_y = 0, output_z = 0
    cdef FLOAT normalizer = 0

    cdef Py_ssize_t rows = nums.shape[0], cols = nums.shape[1], dims = nums.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, num_threads=num_threads, nogil=True):
        for j in prange(cols):
            for k in prange(dims):
                normalizer += contiguous_nums[i, j, k]

                output_x += contiguous_nums[i, j, k] * i
                output_y += contiguous_nums[i, j, k] * j
                output_z += contiguous_nums[i, j, k] * k

    return np.array([output_x / normalizer, output_y / normalizer, output_z / normalizer])
