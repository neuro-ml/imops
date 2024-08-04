# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=True
# cython: overflowcheck.fold=False
# cython: embedsignature=False
# cython: embedsignature.format=c
# cython: c_api_binop_methods=True
# cython: profile=False
# cython: linetrace=False
# cython: infer_types=False
# cython: language_level=3
# cython: cpp_locals=False


from libcpp.unordered_set cimport unordered_set

import numpy as np

cimport numpy as np

from cython.parallel import prange


ctypedef fused NUM:
    short
    int
    long long


cpdef void _isin(const NUM[:] elements, const NUM[:] test_elements, np.uint8_t[:] res, Py_ssize_t num_threads):
    cdef unordered_set[NUM] test_elements_set
    cdef Py_ssize_t i
    cdef Py_ssize_t elements_len = elements.shape[0]
    cdef Py_ssize_t test_elements_len = test_elements.shape[0]

    test_elements_set.reserve(test_elements_len)

    with nogil:
        for i in range(test_elements_len):
            test_elements_set.insert(test_elements[i])

        for i in prange(elements_len, num_threads=num_threads):
            res[i] = test_elements_set.count(elements[i])
