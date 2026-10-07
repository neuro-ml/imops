# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport numpy as np

from cython.parallel import prange

from libc.math cimport floor


ctypedef fused NUM:
    np.float32_t
    np.uint8_t
    np.uint16_t
    np.int16_t
    np.int32_t


cdef inline NUM sample_bilinear(const NUM* plane, Py_ssize_t rows, Py_ssize_t cols,
                                double r, double c, NUM cval) noexcept nogil:
    if r < 0 or r > rows - 1 or c < 0 or c > cols - 1:
        return cval

    cdef Py_ssize_t r0 = <Py_ssize_t>floor(r), c0 = <Py_ssize_t>floor(c)
    cdef Py_ssize_t r1 = r0 + 1 if r0 + 1 < rows else r0
    cdef Py_ssize_t c1 = c0 + 1 if c0 + 1 < cols else c0
    cdef double dr = r - r0, dc = c - c0
    cdef double value = (
        (plane[r0 * cols + c0] * (1 - dc) + plane[r0 * cols + c1] * dc) * (1 - dr) +
        (plane[r1 * cols + c0] * (1 - dc) + plane[r1 * cols + c1] * dc) * dr
    )

    if NUM is np.float32_t:
        return <NUM>value

    return <NUM>(value + 0.5 if value > 0 else value - 0.5)


cdef inline NUM sample_nearest(const NUM* plane, Py_ssize_t rows, Py_ssize_t cols,
                               double r, double c, NUM cval) noexcept nogil:
    if r < 0 or r > rows - 1 or c < 0 or c > cols - 1:
        return cval

    return plane[(<Py_ssize_t>floor(r + 0.5)) * cols + <Py_ssize_t>floor(c + 0.5)]


def _rotate3d_linear(const NUM[:, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                     Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, Py_ssize_t num_threads):
    cdef Py_ssize_t planes = input.shape[0], rows = input.shape[1], cols = input.shape[2]
    cdef NUM[:, :, ::1] rotated = np.empty_like(input, shape=(planes, out_rows, out_cols))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t line, plane, i, j

    for line in prange(planes * out_rows, nogil=True, num_threads=num_threads):
        plane = line / out_rows
        i = line % out_rows
        for j in range(out_cols):
            rotated[plane, i, j] = sample_bilinear(
                &input[plane, 0, 0], rows, cols,
                row_0 + row_i * i + row_j * j,
                col_0 + col_i * i + col_j * j,
                cval,
            )

    return np.asarray(rotated)


def _rotate3d_nearest(const NUM[:, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                      Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, Py_ssize_t num_threads):
    cdef Py_ssize_t planes = input.shape[0], rows = input.shape[1], cols = input.shape[2]
    cdef NUM[:, :, ::1] rotated = np.empty_like(input, shape=(planes, out_rows, out_cols))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t line, plane, i, j

    for line in prange(planes * out_rows, nogil=True, num_threads=num_threads):
        plane = line / out_rows
        i = line % out_rows
        for j in range(out_cols):
            rotated[plane, i, j] = sample_nearest(
                &input[plane, 0, 0], rows, cols,
                row_0 + row_i * i + row_j * j,
                col_0 + col_i * i + col_j * j,
                cval,
            )

    return np.asarray(rotated)
