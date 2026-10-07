# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport numpy as np

from cython.parallel import prange


ctypedef fused NUM:
    np.float32_t
    np.uint8_t
    np.uint16_t
    np.int16_t
    np.int32_t


cdef struct Span:
    Py_ssize_t lo
    Py_ssize_t hi


cdef inline bint outside(double r, double c, Py_ssize_t rows, Py_ssize_t cols) noexcept nogil:
    return r < 0 or r > rows - 1 or c < 0 or c > cols - 1


cdef inline Py_ssize_t clamped(double value, Py_ssize_t low, Py_ssize_t high) noexcept nogil:
    if value <= low:
        return low
    if value >= high:
        return high

    return <Py_ssize_t>value


cdef inline void narrow(double at_column_zero, double per_column, double limit, Span* span) noexcept nogil:
    """Drop the j where `at_column_zero + per_column * j` leaves [0, limit], erring wide by one."""
    cdef double first, last

    if per_column == 0:
        if at_column_zero < 0 or at_column_zero > limit:
            span.hi = span.lo
        return

    first = -at_column_zero / per_column
    last = (limit - at_column_zero) / per_column
    if per_column < 0:
        first, last = last, first

    span.lo = clamped(first, span.lo, span.hi)
    span.hi = clamped(last + 2, span.lo, span.hi)


cdef inline Span inside_span(double row_at_j0, double row_j, Py_ssize_t rows,
                             double col_at_j0, double col_j, Py_ssize_t cols,
                             Py_ssize_t out_cols) noexcept nogil:
    """The j whose source point lies in the plane, so both coordinates are non negative there."""
    cdef Span span

    span.lo = 0
    span.hi = out_cols
    narrow(row_at_j0, row_j, rows - 1, &span)
    narrow(col_at_j0, col_j, cols - 1, &span)

    while span.lo < span.hi and outside(row_at_j0 + row_j * span.lo, col_at_j0 + col_j * span.lo, rows, cols):
        span.lo = span.lo + 1
    while span.hi > span.lo and outside(
        row_at_j0 + row_j * (span.hi - 1), col_at_j0 + col_j * (span.hi - 1), rows, cols
    ):
        span.hi = span.hi - 1

    return span


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


# TODO: tile the (i, j) loops. A shorter j run keeps the source rows in cache, which is what
# the plane (0, 2) layout needs: there the rows sit a whole image row apart.
def _rotate_pixels_linear(const NUM[:, :, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                          Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, bint mask, Py_ssize_t num_threads):
    cdef Py_ssize_t pre = input.shape[0], rows = input.shape[1], mid = input.shape[2], cols = input.shape[3]
    cdef NUM[:, :, :, ::1] rotated = np.empty_like(input, shape=(pre, out_rows, mid, out_cols))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t row_stride = mid * cols
    cdef Py_ssize_t line, p, i, m, j, r0, c0, row_step, col_step
    cdef double r, c, row_at_j0, col_at_j0
    cdef Span span
    cdef const NUM *plane
    cdef NUM *out_row

    for line in prange(pre * mid * out_rows, nogil=True, num_threads=num_threads):
        p = line / (mid * out_rows)
        m = line / out_rows % mid
        i = line % out_rows
        plane = &input[p, 0, m, 0]
        out_row = &rotated[p, i, m, 0]
        row_at_j0 = row_0 + row_i * i
        col_at_j0 = col_0 + col_i * i
        span = inside_span(row_at_j0, row_j, rows, col_at_j0, col_j, cols, out_cols)

        for j in range(span.lo):
            out_row[j] = cval

        for j in range(span.lo, span.hi):
            r = row_at_j0 + row_j * j
            c = col_at_j0 + col_j * j
            r0 = <Py_ssize_t>r
            c0 = <Py_ssize_t>c
            row_step = row_stride if r0 + 1 < rows else 0
            col_step = 1 if c0 + 1 < cols else 0
            store(out_row + j, blend(plane + r0 * row_stride + c0, row_step, col_step, r - r0, c - c0), mask)

        for j in range(span.hi, out_cols):
            out_row[j] = cval

    return np.asarray(rotated)


def _rotate_pixels_nearest(const NUM[:, :, :, ::1] input, double[:, ::1] matrix, double[::1] shift,
                           Py_ssize_t out_rows, Py_ssize_t out_cols, NUM cval, Py_ssize_t num_threads):
    cdef Py_ssize_t pre = input.shape[0], rows = input.shape[1], mid = input.shape[2], cols = input.shape[3]
    cdef NUM[:, :, :, ::1] rotated = np.empty_like(input, shape=(pre, out_rows, mid, out_cols))

    cdef double row_i = matrix[0, 0], row_j = matrix[0, 1], row_0 = shift[0]
    cdef double col_i = matrix[1, 0], col_j = matrix[1, 1], col_0 = shift[1]
    cdef Py_ssize_t row_stride = mid * cols
    cdef Py_ssize_t line, p, i, m, j
    cdef double row_at_j0, col_at_j0
    cdef Span span
    cdef const NUM *plane
    cdef NUM *out_row

    for line in prange(pre * mid * out_rows, nogil=True, num_threads=num_threads):
        p = line / (mid * out_rows)
        m = line / out_rows % mid
        i = line % out_rows
        plane = &input[p, 0, m, 0]
        out_row = &rotated[p, i, m, 0]
        row_at_j0 = row_0 + row_i * i
        col_at_j0 = col_0 + col_i * i
        span = inside_span(row_at_j0, row_j, rows, col_at_j0, col_j, cols, out_cols)

        for j in range(span.lo):
            out_row[j] = cval

        for j in range(span.lo, span.hi):
            out_row[j] = plane[
                (<Py_ssize_t>(row_at_j0 + row_j * j + 0.5)) * row_stride + <Py_ssize_t>(col_at_j0 + col_j * j + 0.5)
            ]

        for j in range(span.hi, out_cols):
            out_row[j] = cval

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
    cdef double r, c, dr, dc, row_at_j0, col_at_j0
    cdef Span span
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
        row_at_j0 = row_0 + row_i * i
        col_at_j0 = col_0 + col_i * i
        span = inside_span(row_at_j0, row_j, rows, col_at_j0, col_j, cols, out_cols)

        for t in range(span.lo * post):
            out_base[t] = cval

        for j in range(span.lo, span.hi):
            r = row_at_j0 + row_j * j
            c = col_at_j0 + col_j * j
            r0 = <Py_ssize_t>r
            c0 = <Py_ssize_t>c
            row_step = row_stride if r0 + 1 < rows else 0
            col_step = post if c0 + 1 < cols else 0
            dr = r - r0
            dc = c - c0
            tap = plane + r0 * row_stride + c0 * post
            out_run = out_base + j * post

            for t in range(post):
                store(out_run + t, blend(tap + t, row_step, col_step, dr, dc), mask)

        for t in range(span.hi * post, out_cols * post):
            out_base[t] = cval

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
    cdef double row_at_j0, col_at_j0
    cdef Span span
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
        row_at_j0 = row_0 + row_i * i
        col_at_j0 = col_0 + col_i * i
        span = inside_span(row_at_j0, row_j, rows, col_at_j0, col_j, cols, out_cols)

        for t in range(span.lo * post):
            out_base[t] = cval

        for j in range(span.lo, span.hi):
            tap = plane + (<Py_ssize_t>(row_at_j0 + row_j * j + 0.5)) * row_stride + (
                <Py_ssize_t>(col_at_j0 + col_j * j + 0.5)
            ) * post
            out_run = out_base + j * post

            for t in range(post):
                out_run[t] = tap[t]

        for t in range(span.hi * post, out_cols * post):
            out_base[t] = cval

    return np.asarray(rotated)
