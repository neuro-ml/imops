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


ctypedef cython.numeric NUM


cdef inline NUM _sum(NUM[:] nums) nogil:
    cdef Py_ssize_t i = 0
    cdef NUM sum_ = 0

    for i in range(len(nums)):
        sum_ += nums[i]

    return sum_


def _parallel_sum(NUM[:] nums, Py_ssize_t num_threads) -> NUM:
    cdef Py_ssize_t nums_len = len(nums)
    cdef NUM[:] subsums = np.empty_like(nums, shape=num_threads)
    cdef Py_ssize_t chunksize = nums_len // num_threads
    cdef Py_ssize_t i

    for i in prange(num_threads, nogil=True, num_threads=num_threads):
        subsums[i] = _sum(nums[i * chunksize: (i + 1) * chunksize])

    return np.sum(subsums) + np.sum(nums[chunksize * num_threads:])


def _parallel_pointwise_mul(NUM[:] nums1, NUM[:] nums2, Py_ssize_t num_threads) -> np.ndarray:
    cdef Py_ssize_t nums1_len = len(nums1)
    cdef NUM[:] mul = np.empty_like(nums1, shape=nums1_len)
    cdef Py_ssize_t i

    for i in prange(nums1_len, nogil=True, num_threads=num_threads):
        mul[i] = nums1[i] * nums2[i]

    return np.asarray(mul)
