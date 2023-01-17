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


# TODO: Move generic functions like this to the separate file
cdef inline np.uint8_t get_pixel3d(np.uint8_t* input,
                                   Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                                   Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                                   np.uint8_t cval, layout = b'C') nogil:
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols) or (d < 0) or (d >= dims):
            return cval

        if layout == b'C':
            return input[r * cols * dims + c * dims + d]

        if layout == b'F':
            return input[rows * cols * d + rows * c + r]


cdef inline np.uint8_t max_in_footprint(np.uint8_t* input, np.uint8_t* footprint,
                     Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                     Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
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
                        i, j, k,
                        False,
                    ) and
                    get_pixel3d(
                        input,
                        rows, cols, dims,
                        i_r, i_c, i_d,
                        False,
                    )
                ):
                    return True

    return False


cdef inline np.uint8_t min_in_footprint(np.uint8_t* input, np.uint8_t* footprint,
                     Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                     Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
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
                        i, j, k,
                        False,
                    ) and not
                    get_pixel3d(
                        input,
                        rows, cols, dims,
                        i_r, i_c, i_d,
                        True,
                    )
                ):
                    return False

    return True


def _binary_dilation(np.uint8_t[:, :, :] input, np.uint8_t[:, :, :] footprint, Py_ssize_t num_threads):
    cdef np.uint8_t[:, :, ::1] contiguous_input = np.ascontiguousarray(input)
    cdef np.uint8_t[:, :, ::1] contiguous_footprint = np.ascontiguousarray(footprint)

    cdef np.uint8_t[:, :, ::1] dilated = np.zeros_like(input, dtype=np.uint8)

    cdef Py_ssize_t rows = input.shape[0], cols = input.shape[1], dims = input.shape[2]
    cdef Py_ssize_t f_rows = footprint.shape[0], f_cols = footprint.shape[1], f_dims = footprint.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                dilated[i, j, k] = max_in_footprint(
                    &contiguous_input[0, 0, 0],
                    &contiguous_footprint[0, 0, 0],
                    i, j, k, rows, cols, dims, f_rows, f_cols, f_dims
                )

    return np.asarray(dilated)


def _binary_erosion(np.uint8_t[:, :, :] input, np.uint8_t[:, :, :] footprint, Py_ssize_t num_threads):
    cdef np.uint8_t[:, :, ::1] contiguous_input = np.ascontiguousarray(input)
    cdef np.uint8_t[:, :, ::1] contiguous_footprint = np.ascontiguousarray(footprint)

    cdef np.uint8_t[:, :, ::1] eroded = np.zeros_like(input, dtype=np.uint8)

    cdef Py_ssize_t rows = input.shape[0], cols = input.shape[1], dims = input.shape[2]
    cdef Py_ssize_t f_rows = footprint.shape[0], f_cols = footprint.shape[1], f_dims = footprint.shape[2]
    cdef Py_ssize_t i, j, k

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                eroded[i, j, k] = min_in_footprint(
                    &contiguous_input[0, 0, 0],
                    &contiguous_footprint[0, 0, 0],
                    i, j, k, rows, cols, dims, f_rows, f_cols, f_dims
                )

    return np.asarray(eroded)
