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


def _inner_argmax(const float[:, :] x, Py_ssize_t argmax_dim, Py_ssize_t post_dim, Py_ssize_t num_threads):
    cdef float tmp, tmp_max
    cdef unsigned char cmax
    cdef Py_ssize_t i, c
    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=1, mode="c"] out = np.empty((post_dim, ), dtype=np.uint8)

    for i in prange(0, post_dim, nogil=True, num_threads=num_threads):
        tmp_max = x[0, i]
        cmax = 0

        for c in range(1, argmax_dim):
            tmp = x[c, i]

            if tmp > tmp_max:
                cmax = c
                tmp_max = tmp

        out[i] = cmax

    return out


def _inner_argmax_out(const float[:, :] x, unsigned char[:] out, Py_ssize_t argmax_dim, Py_ssize_t post_dim, Py_ssize_t num_threads):
    cdef float tmp, tmp_max
    cdef unsigned char cmax
    cdef Py_ssize_t i, c

    for i in prange(0, post_dim, nogil=True, num_threads=num_threads):
        tmp_max = x[0, i]
        cmax = 0

        for c in range(1, argmax_dim):
            tmp = x[c, i]

            if tmp > tmp_max:
                cmax = c
                tmp_max = tmp

        out[i] = cmax


def _outer_argmax(const float[:, :] x, Py_ssize_t argmax_dim, Py_ssize_t pre_dim, Py_ssize_t num_threads):
    cdef float tmp, tmp_max
    cdef unsigned char cmax
    cdef Py_ssize_t i, c
    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=1, mode="c"] out = np.empty((pre_dim, ), dtype=np.uint8)

    for i in prange(0, pre_dim, nogil=True, num_threads=num_threads):
        tmp_max = x[i, 0]
        cmax = 0

        for c in range(1, argmax_dim):
            tmp = x[i, c]

            if tmp > tmp_max:
                cmax = c
                tmp_max = tmp

        out[i] = cmax

    return out


def _outer_inner_argmax(const float[:, :, :] x, Py_ssize_t pre_dim, Py_ssize_t argmax_dim, Py_ssize_t post_dim, Py_ssize_t num_threads):
    cdef float tmp, tmp_max
    cdef unsigned char cmax
    cdef Py_ssize_t i, c, j
    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=2, mode="c"] out = np.empty((pre_dim, post_dim), dtype=np.uint8)

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
