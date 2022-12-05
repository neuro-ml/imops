# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

cimport cython
cimport numpy as np


cdef inline BOOL_FP32_FP64 get_pixel2d(BOOL_FP32_FP64* input,
                                       Py_ssize_t rows, Py_ssize_t cols,
                                       Py_ssize_t r, Py_ssize_t c,
                                       BOOL_FP32_FP64 cval) nogil:
    if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
        return cval

    return input[r * cols + c]


cdef inline BOOL_FP32_FP64 get_pixel3d(BOOL_FP32_FP64* input,
                                       Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                                       Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                                       BOOL_FP32_FP64 cval) nogil:
    if (r < 0) or (r >= rows) or (c < 0) or (c >= cols) or (d < 0) or (d >= dims):
        return cval

    return input[r * cols * dims + c * dims + d]
