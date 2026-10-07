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


cdef inline double source_coordinate(double constant, double per_row, double per_col,
                                     Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    return constant + per_row * i + per_col * j


cdef inline bint outside(double r, double c, Py_ssize_t rows, Py_ssize_t cols) noexcept nogil:
    return r < 0 or r > rows - 1 or c < 0 or c > cols - 1


cdef inline Py_ssize_t nearest_offset(double r, double c,
                                      Py_ssize_t row_stride, Py_ssize_t col_stride) noexcept nogil:
    return (<Py_ssize_t>floor(r + 0.5)) * row_stride + (<Py_ssize_t>floor(c + 0.5)) * col_stride


cdef inline double blend(const NUM* tap, Py_ssize_t row_step, Py_ssize_t col_step,
                         double dr, double dc) noexcept nogil:
    return (
        (tap[0] * (1 - dc) + tap[col_step] * dc) * (1 - dr) +
        (tap[row_step] * (1 - dc) + tap[row_step + col_step] * dc) * dr
    )


cdef inline void store(NUM* out, double value, bint mask) noexcept nogil:
    if NUM is np.float32_t:
        out[0] = <NUM>value
    elif mask:
        out[0] = <NUM>((<float>value) >= 0.5)
    else:
        out[0] = <NUM>(value + 0.5 if value > 0 else value - 0.5)


def _rotate_pixels_linear(const NUM[:, :, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                          Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, bint mask, Py_ssize_t num_threads):
    cdef Py_ssize_t pre = input.shape[0], rows = input.shape[1], mid = input.shape[2], cols = input.shape[3]
    cdef NUM[:, :, :, ::1] rotated = np.empty_like(input, shape=(pre, out_rows, mid, out_cols))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t row_stride = mid * cols
    cdef Py_ssize_t line, p, i, m, j, r0, c0, row_step, col_step
    cdef double r, c
    cdef const NUM *plane
    cdef NUM *out_row

    for line in prange(pre * mid * out_rows, nogil=True, num_threads=num_threads):
        p = line / (mid * out_rows)
        m = line / out_rows % mid
        i = line % out_rows
        plane = &input[p, 0, m, 0]
        out_row = &rotated[p, i, m, 0]

        for j in range(out_cols):
            r = source_coordinate(row_0, row_i, row_j, i, j)
            c = source_coordinate(col_0, col_i, col_j, i, j)

            if outside(r, c, rows, cols):
                out_row[j] = cval
                continue

            r0 = <Py_ssize_t>floor(r)
            c0 = <Py_ssize_t>floor(c)
            row_step = row_stride if r0 + 1 < rows else 0
            col_step = 1 if c0 + 1 < cols else 0
            store(out_row + j, blend(plane + r0 * row_stride + c0, row_step, col_step, r - r0, c - c0), mask)

    return np.asarray(rotated)


def _rotate_pixels_nearest(const NUM[:, :, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                           Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, Py_ssize_t num_threads):
    cdef Py_ssize_t pre = input.shape[0], rows = input.shape[1], mid = input.shape[2], cols = input.shape[3]
    cdef NUM[:, :, :, ::1] rotated = np.empty_like(input, shape=(pre, out_rows, mid, out_cols))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t row_stride = mid * cols
    cdef Py_ssize_t line, p, i, m, j
    cdef double r, c
    cdef const NUM *plane
    cdef NUM *out_row

    for line in prange(pre * mid * out_rows, nogil=True, num_threads=num_threads):
        p = line / (mid * out_rows)
        m = line / out_rows % mid
        i = line % out_rows
        plane = &input[p, 0, m, 0]
        out_row = &rotated[p, i, m, 0]

        for j in range(out_cols):
            r = source_coordinate(row_0, row_i, row_j, i, j)
            c = source_coordinate(col_0, col_i, col_j, i, j)

            if outside(r, c, rows, cols):
                out_row[j] = cval
            else:
                out_row[j] = plane[nearest_offset(r, c, row_stride, 1)]

    return np.asarray(rotated)


def _rotate_runs_linear(const NUM[:, :, :, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                        Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, bint mask, Py_ssize_t num_threads):
    cdef Py_ssize_t pre = input.shape[0], rows = input.shape[1], mid = input.shape[2]
    cdef Py_ssize_t cols = input.shape[3], post = input.shape[4]
    cdef NUM[:, :, :, :, ::1] rotated = np.empty_like(input, shape=(pre, out_rows, mid, out_cols, post))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t row_stride = mid * cols * post
    cdef Py_ssize_t line, p, i, m, j, t, r0, c0, row_step, col_step
    cdef double r, c, dr, dc
    cdef const NUM *plane
    cdef const NUM *tap
    cdef NUM *out_base
    cdef NUM *out_run

    for line in prange(pre * mid * out_rows, nogil=True, num_threads=num_threads):
        p = line / (mid * out_rows)
        m = line / out_rows % mid
        i = line % out_rows
        plane = &input[p, 0, m, 0, 0]
        out_base = &rotated[p, i, m, 0, 0]

        for j in range(out_cols):
            r = source_coordinate(row_0, row_i, row_j, i, j)
            c = source_coordinate(col_0, col_i, col_j, i, j)
            out_run = out_base + j * post

            if outside(r, c, rows, cols):
                for t in range(post):
                    out_run[t] = cval
                continue

            r0 = <Py_ssize_t>floor(r)
            c0 = <Py_ssize_t>floor(c)
            row_step = row_stride if r0 + 1 < rows else 0
            col_step = post if c0 + 1 < cols else 0
            dr = r - r0
            dc = c - c0
            tap = plane + r0 * row_stride + c0 * post

            for t in range(post):
                store(out_run + t, blend(tap + t, row_step, col_step, dr, dc), mask)

    return np.asarray(rotated)


def _rotate_runs_nearest(const NUM[:, :, :, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                         Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, Py_ssize_t num_threads):
    cdef Py_ssize_t pre = input.shape[0], rows = input.shape[1], mid = input.shape[2]
    cdef Py_ssize_t cols = input.shape[3], post = input.shape[4]
    cdef NUM[:, :, :, :, ::1] rotated = np.empty_like(input, shape=(pre, out_rows, mid, out_cols, post))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t row_stride = mid * cols * post
    cdef Py_ssize_t line, p, i, m, j, t
    cdef double r, c
    cdef const NUM *plane
    cdef const NUM *tap
    cdef NUM *out_base
    cdef NUM *out_run

    for line in prange(pre * mid * out_rows, nogil=True, num_threads=num_threads):
        p = line / (mid * out_rows)
        m = line / out_rows % mid
        i = line % out_rows
        plane = &input[p, 0, m, 0, 0]
        out_base = &rotated[p, i, m, 0, 0]

        for j in range(out_cols):
            r = source_coordinate(row_0, row_i, row_j, i, j)
            c = source_coordinate(col_0, col_i, col_j, i, j)
            out_run = out_base + j * post

            if outside(r, c, rows, cols):
                for t in range(post):
                    out_run[t] = cval
                continue

            tap = plane + nearest_offset(r, c, row_stride, post)

            for t in range(post):
                out_run[t] = tap[t]

    return np.asarray(rotated)
