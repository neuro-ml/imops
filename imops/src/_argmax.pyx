# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np

cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport ceilf, floorf

cnp.import_array()


def _first_index_4d_argmax(const FLOAT[:,:,:,:] x, Py_ssize_t C, Py_ssize_t I, Py_ssize_t J, Py_ssize_t K, Py_ssize_t num_threads):
    cdef Py_ssize_t c, i, j, k
    cdef FLOAT tmp

    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=3, mode="c"] out = np.zeros((I, J, K), dtype=np.uint8)
    cdef cnp.ndarray[dtype=cnp.float32, ndim=3, mode="c"] tmp_max = np.copy(x[0])

    for c in prange(1, C, nogil=True, num_threads=num_threads)
        for i in prange(0, I):
            for j in prange(0, J):
                for k in prange(0, K):
                    tmp = x[c, i, j, k]

                    if tmp > tmp_max[i, j, k]:
                        out[i, j, k] = c
                        tmp_max[i, j, k] = tmp

    return out
