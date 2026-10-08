# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np
from cython.parallel import prange

cimport cython
cimport numpy as np
from cython.parallel cimport threadid


ctypedef cython.floating FLOAT
ctypedef np.uint8_t uint8

cdef enum:
    # Output pixels along a tile side. A tile and the detector samples it reads have to share the L1 cache.
    TILE = 16
    # Slices and angles swapped at once, one cache line each way.
    BLOCK = 16


cdef struct Span:
    Py_ssize_t lo
    Py_ssize_t hi


cdef inline Py_ssize_t clamped(Py_ssize_t value, Py_ssize_t low, Py_ssize_t high) noexcept nogil:
    if value < low:
        return low
    if value > high:
        return high

    return value


cdef inline Span inside_span(const uint8[:] row) noexcept nogil:
    """The circle meets a row of the output in one interval, so its two ends are all the kernel needs."""
    cdef Span span

    span.lo = 0
    span.hi = row.shape[0]
    while span.lo < span.hi and not row[span.lo]:
        span.lo = span.lo + 1
    while span.hi > span.lo and not row[span.hi - 1]:
        span.hi = span.hi - 1

    return span


cdef inline Span overlap(Py_ssize_t lo, Py_ssize_t hi, Py_ssize_t left, Py_ssize_t right) noexcept nogil:
    """Where a row's circle interval meets the columns of a tile. A miss collapses, it never inverts."""
    cdef Span span

    span.lo = clamped(lo, left, right)
    span.hi = clamped(hi, span.lo, right)

    return span


cdef inline void add_sample(FLOAT* run, const FLOAT* tap, Py_ssize_t ahead, FLOAT weight,
                            Py_ssize_t n_slices) noexcept nogil:
    """One detector sample interpolated into every slice at once. `ahead` is 0 at the last sample, which weighs 0."""
    cdef Py_ssize_t s

    for s in range(n_slices):
        run[s] = run[s] + tap[s] + (tap[ahead + s] - tap[s]) * weight


cdef inline void fill(FLOAT* run, FLOAT value, Py_ssize_t length) noexcept nogil:
    cdef Py_ssize_t s

    for s in range(length):
        run[s] = value


def by_angle(const FLOAT[:, :, ::1] sinogram, Py_ssize_t image_size, Py_ssize_t offset, Py_ssize_t num_threads):
    """The sinogram as (angle, detector, slice), its detector centred in one `image_size` long."""
    cdef Py_ssize_t n_slices = sinogram.shape[0], detector = sinogram.shape[1], n_angles = sinogram.shape[2]
    cdef FLOAT[:, :, ::1] rows = np.zeros_like(sinogram, shape=(n_angles, image_size, n_slices), order='C')
    cdef Py_ssize_t angle_blocks = (n_angles + BLOCK - 1) // BLOCK
    cdef Py_ssize_t d, angle_block, k0, k_end, s, k

    for d in prange(detector, nogil=True, num_threads=num_threads):
        for angle_block in range(angle_blocks):
            k0 = angle_block * BLOCK
            k_end = clamped(k0 + BLOCK, k0, n_angles)

            for s in range(n_slices):
                for k in range(k0, k_end):
                    rows[k, offset + d, s] = sinogram[s, d, k]

    return np.asarray(rows)


cpdef FLOAT[:, :, :] backprojection3d(const FLOAT[:, :, ::1] rows, const FLOAT[:] theta, const FLOAT[:] xs,
                                      const uint8[:, :] inside_circle, FLOAT fill_value, int image_size,
                                      Py_ssize_t num_threads):
    """`rows` is the filtered sinogram as (angle, detector, slice); only its first `image_size` samples are read."""
    cdef Py_ssize_t n_angles = theta.shape[0], n_slices = rows.shape[2]
    cdef Py_ssize_t output_size = inside_circle.shape[0]
    cdef FLOAT[:, :, ::1] result = np.empty_like(rows, shape=(n_slices, output_size, output_size), order='C')
    cdef FLOAT[:, ::1] patches = np.empty_like(rows, shape=(num_threads, TILE * TILE * n_slices), order='C')
    cdef Py_ssize_t[:, ::1] spans = np.empty((output_size, 2), dtype=np.intp)
    cdef FLOAT[:] sinuses = np.sin(theta)
    cdef FLOAT[:] cosinuses = np.cos(theta)
    cdef FLOAT shift = image_size // 2, right_limit = image_size - 1
    cdef FLOAT multiplier = np.pi / (2 * n_angles)
    cdef Py_ssize_t across = (output_size + TILE - 1) // TILE
    cdef Py_ssize_t tile, top, bottom, left, right, i, j, k, s, idx
    cdef FLOAT value, start
    cdef FLOAT* patch
    cdef Span span

    for i in range(output_size):
        span = inside_span(inside_circle[i])
        spans[i, 0] = span.lo
        spans[i, 1] = span.hi

    for tile in prange(across * across, nogil=True, num_threads=num_threads):
        top = tile / across * TILE
        left = tile % across * TILE
        bottom = clamped(top + TILE, top, output_size)
        right = clamped(left + TILE, left, output_size)
        patch = &patches[threadid(), 0]
        fill(patch, 0, (bottom - top) * TILE * n_slices)

        for k in range(n_angles):
            for i in range(top, bottom):
                span = overlap(spans[i, 0], spans[i, 1], left, right)
                start = -xs[i] * sinuses[k]

                for j in range(span.lo, span.hi):
                    value = xs[j] * cosinuses[k] + start + shift
                    if value < 0 or value > right_limit:
                        continue

                    idx = <Py_ssize_t>value
                    add_sample(
                        patch + ((i - top) * TILE + j - left) * n_slices, &rows[k, idx, 0],
                        n_slices if idx < right_limit else 0, value - idx, n_slices
                    )

        for s in range(n_slices):
            for i in range(top, bottom):
                span = overlap(spans[i, 0], spans[i, 1], left, right)

                for j in range(left, span.lo):
                    result[s, i, j] = fill_value

                for j in range(span.lo, span.hi):
                    result[s, i, j] = patch[((i - top) * TILE + j - left) * n_slices + s] * multiplier

                for j in range(span.hi, right):
                    result[s, i, j] = fill_value

    return np.asarray(result)
