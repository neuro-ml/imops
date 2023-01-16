# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport cython
cimport numpy as np

from cython import nogil
from cython.parallel import prange

from ._utils cimport BOOL, get_pixel3d


cdef inline BOOL max_in_footprint(BOOL* input, BOOL* footprint,
                                  Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                                  Py_ssize_t stride_r, Py_ssize_t stride_c, Py_ssize_t stride_d,
                                  Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                                  Py_ssize_t f_stride_r, Py_ssize_t f_stride_c, Py_ssize_t f_stride_d,
                                  Py_ssize_t f_rows, Py_ssize_t f_cols, Py_ssize_t f_dims) nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t i_r, i_c, i_d

    cdef int rows_shift = r + f_rows // 2
    cdef int cols_shift = c + f_cols // 2
    cdef int dims_shift = d + f_dims // 2

    for i in range(f_rows):
        i_r = rows_shift - i
        for j in range(f_cols):
            i_c = cols_shift - j
            for k in range(f_dims):
                i_d = dims_shift - k

                if (
                    get_pixel3d(
                        footprint,
                        f_rows, f_cols, f_dims,
                        f_stride_r, f_stride_c, f_stride_d,
                        i, j, k,
                        False,
                    ) and
                    get_pixel3d(
                        input,
                        rows, cols, dims,
                        stride_r, stride_c, stride_d,
                        i_r, i_c, i_d,
                        False,
                    )
                ):
                    return True

    return False


cdef inline BOOL min_in_footprint(BOOL* input, BOOL* footprint,
                                  Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                                  Py_ssize_t stride_r, Py_ssize_t stride_c, Py_ssize_t stride_d,
                                  Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                                  Py_ssize_t f_stride_r, Py_ssize_t f_stride_c, Py_ssize_t f_stride_d,
                                  Py_ssize_t f_rows, Py_ssize_t f_cols, Py_ssize_t f_dims) nogil:
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t i_r, i_c, i_d

    cdef int rows_shift = r - f_rows // 2
    cdef int cols_shift = c - f_cols // 2
    cdef int dims_shift = d - f_dims // 2

    for i in range(f_rows):
        i_r = rows_shift + i
        for j in range(f_cols):
            i_c = cols_shift + j
            for k in range(f_dims):
                i_d = dims_shift + k

                if (
                    get_pixel3d(
                        footprint,
                        f_rows, f_cols, f_dims,
                        f_stride_r, f_stride_c, f_stride_d,
                        i, j, k,
                        False,
                    ) and not
                    get_pixel3d(
                        input,
                        rows, cols, dims,
                        stride_r, stride_c, stride_d,
                        i_r, i_c, i_d,
                        True,
                    )
                ):
                    return False

    return True


def _binary_dilation(BOOL[:, :, :] input, BOOL[:, :, :] footprint, Py_ssize_t num_threads):
    cdef BOOL[:, :, ::1] contiguous_input = np.ascontiguousarray(input)
    cdef BOOL[:, :, ::1] contiguous_footprint = np.ascontiguousarray(footprint)

    cdef BOOL[:, :, ::1] dilated = np.zeros_like(input, dtype=np.uint8)

    cdef Py_ssize_t rows = input.shape[0], cols = input.shape[1], dims = input.shape[2]
    cdef Py_ssize_t f_rows = footprint.shape[0], f_cols = footprint.shape[1], f_dims = footprint.shape[2]
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t stride_r = cols * dims, stride_c = dims, stride_d = 1
    cdef Py_ssize_t f_stride_r = f_cols * f_dims, f_stride_c = f_dims, f_stride_d = 1

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                dilated[i, j, k] = max_in_footprint(
                    &contiguous_input[0, 0, 0],
                    &contiguous_footprint[0, 0, 0],
                    i, j, k,
                    stride_r, stride_c, stride_d,
                    rows, cols, dims,
                    f_stride_r, f_stride_c, f_stride_d,
                    f_rows, f_cols, f_dims,
                )

    return np.asarray(dilated)


def _binary_erosion(BOOL[:, :, :] input, BOOL[:, :, :] footprint, Py_ssize_t num_threads):
    cdef BOOL[:, :, ::1] contiguous_input = np.ascontiguousarray(input)
    cdef BOOL[:, :, ::1] contiguous_footprint = np.ascontiguousarray(footprint)

    cdef BOOL[:, :, ::1] eroded = np.zeros_like(input, dtype=np.uint8)

    cdef Py_ssize_t rows = input.shape[0], cols = input.shape[1], dims = input.shape[2]
    cdef Py_ssize_t f_rows = footprint.shape[0], f_cols = footprint.shape[1], f_dims = footprint.shape[2]
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t stride_r = cols * dims, stride_c = dims, stride_d = 1
    cdef Py_ssize_t f_stride_r = f_cols * f_dims, f_stride_c = f_dims, f_stride_d = 1

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                eroded[i, j, k] = min_in_footprint(
                    &contiguous_input[0, 0, 0],
                    &contiguous_footprint[0, 0, 0],
                    i, j, k,
                    stride_r, stride_c, stride_d,
                    rows, cols, dims,
                    f_stride_r, f_stride_c, f_stride_d,
                    f_rows, f_cols, f_dims,
                )

    return np.asarray(eroded)
