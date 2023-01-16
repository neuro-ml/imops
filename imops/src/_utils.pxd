# cython: language_level = 3

cimport cython
cimport numpy as np


ctypedef cython.floating FP32_FP64
ctypedef np.uint8_t BOOL
ctypedef fused BOOL_FP32_FP64:
    BOOL
    FP32_FP64


# FIXME: Why cython throws warning on inline modifier in .pxd?
cdef inline BOOL_FP32_FP64 get_pixel2d(BOOL_FP32_FP64* input,
                                       Py_ssize_t rows, Py_ssize_t cols,
                                       Py_ssize_t r, Py_ssize_t c,
                                       BOOL_FP32_FP64 cval) nogil


cdef inline BOOL_FP32_FP64 get_pixel3d(BOOL_FP32_FP64* input,
                                       Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                                       Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                                       BOOL_FP32_FP64 cval) nogil
