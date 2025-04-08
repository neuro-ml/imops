# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport numpy as cnp

from cython.parallel import prange


cnp.import_array()

ctypedef fused NUM:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    short
    int
    long long
    float
    double


def _argmax(
    const NUM[:, :, :] x,
    Py_ssize_t pre_dim,
    Py_ssize_t argmax_dim,
    Py_ssize_t post_dim,
    Py_ssize_t num_threads
):
    cdef NUM tmp, tmp_max
    cdef unsigned char cmax
    cdef Py_ssize_t i, c, j
    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=2, mode="c"] out = np.empty((pre_dim, post_dim), dtype=np.uint8)

    if pre_dim < num_threads:
        for i in range(pre_dim):
            for j in prange(0, post_dim, nogil=True, num_threads=num_threads):
                tmp_max = x[i, 0, j]
                cmax = 0

                for c in range(1, argmax_dim):
                    tmp = x[i, c, j]

                    if tmp > tmp_max:
                        cmax = c
                        tmp_max = tmp

                out[i, j] = cmax

    elif post_dim < num_threads:
        for i in prange(0, pre_dim, nogil=True, num_threads=num_threads):
            for j in range(post_dim):
                tmp_max = x[i, 0, j]
                cmax = 0

                for c in range(1, argmax_dim):
                    tmp = x[i, c, j]

                    if tmp > tmp_max:
                        cmax = c
                        tmp_max = tmp

                out[i, j] = cmax

    else:
        for i in prange(0, pre_dim, nogil=True, num_threads=num_threads):
            for j in prange(0, post_dim):
                tmp_max = x[i, 0, j]
                cmax = 0

                for c in range(1, argmax_dim):
                    tmp = x[i, c, j]

                    if tmp > tmp_max:
                        cmax = c
                        tmp_max = tmp

                out[i, j] = cmax

    return out
