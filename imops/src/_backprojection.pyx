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
from libc.string cimport memset


ctypedef cython.floating FLOAT
ctypedef np.uint8_t uint8

cdef enum:
    # Output pixels along a tile side. A tile and the detector samples it reads have to share the L1 cache.
    TILE = 16
    # Slices and angles swapped at once, one cache line each way.
    BLOCK = 16


def slices_last(const FLOAT[:, :, ::1] sinogram, Py_ssize_t length, Py_ssize_t num_threads):
    """The sinogram as (angle, detector, slice), its detector padded with zeros to `length`."""
    cdef Py_ssize_t n_slices = sinogram.shape[0], n_samples = sinogram.shape[1], n_angles = sinogram.shape[2]
    cdef FLOAT[:, :, ::1] projections = np.zeros_like(sinogram, shape=(n_angles, length, n_slices), order='C')
    # A block goes through a buffer, so each cache line is read and written whole. Written in place, sixteen
    # lines a power of two apart evict each other, and the copy runs up to ten times slower.
    cdef FLOAT[:, ::1] blocks = np.empty_like(sinogram, shape=(num_threads, BLOCK * BLOCK), order='C')
    cdef Py_ssize_t sample, angle_block, slice_block, first_angle, last_angle, first_slice, last_slice, angle, s
    cdef FLOAT* block

    for sample in prange(n_samples, nogil=True, num_threads=num_threads):
        block = &blocks[threadid(), 0]
        for angle_block in range((n_angles + BLOCK - 1) // BLOCK):
            first_angle = angle_block * BLOCK
            last_angle = min(first_angle + BLOCK, n_angles)
            for slice_block in range((n_slices + BLOCK - 1) // BLOCK):
                first_slice = slice_block * BLOCK
                last_slice = min(first_slice + BLOCK, n_slices)

                for s in range(first_slice, last_slice):
                    for angle in range(first_angle, last_angle):
                        block[(angle - first_angle) * BLOCK + s - first_slice] = sinogram[s, sample, angle]

                for angle in range(first_angle, last_angle):
                    for s in range(first_slice, last_slice):
                        projections[angle, sample, s] = block[(angle - first_angle) * BLOCK + s - first_slice]

    return np.asarray(projections)


cdef inline FLOAT* pixel_sums(FLOAT* tile_sums, Py_ssize_t row, Py_ssize_t col, Py_ssize_t n_slices) noexcept nogil:
    return tile_sums + (row * TILE + col) * n_slices


cdef inline void add_two_angles(FLOAT* sums, const FLOAT* first, FLOAT first_position, const FLOAT* second,
                                FLOAT second_position, Py_ssize_t n_slices) noexcept nogil:
    """Adds to every slice two detector values, each at its position between the two nearest samples."""
    cdef Py_ssize_t first_sample = <Py_ssize_t>first_position, second_sample = <Py_ssize_t>second_position, s
    cdef const FLOAT* a = first + first_sample * n_slices
    cdef const FLOAT* b = second + second_sample * n_slices
    cdef FLOAT a_weight = first_position - first_sample, b_weight = second_position - second_sample

    for s in range(n_slices):
        sums[s] = sums[s] + a[s] + (a[n_slices + s] - a[s]) * a_weight + b[s] + (b[n_slices + s] - b[s]) * b_weight


def backprojection3d(const FLOAT[:, :, ::1] projections, const FLOAT[:] theta, const FLOAT[:] xs,
                     const uint8[:, :] inside_circle, FLOAT fill_value, FLOAT centre, Py_ssize_t num_threads):
    """`projections` is the filtered sinogram as (angle, detector, slice).

    The ray through the pixel at (x, y) meets it at `x * cos - y * sin + centre`, and the sample after is read too.
    """
    cdef Py_ssize_t n_angles = theta.shape[0], n_slices = projections.shape[2], size = inside_circle.shape[0]
    cdef FLOAT[:, :, ::1] image = np.empty_like(projections, shape=(n_slices, size, size), order='C')
    cdef FLOAT[:, ::1] sums = np.empty_like(projections, shape=(num_threads, TILE * TILE * n_slices), order='C')
    # Angles go in pairs, so `sums` is read and written once per two. An odd one out pairs with zeros.
    cdef FLOAT[:, ::1] zeros = np.zeros_like(projections, shape=(projections.shape[1], n_slices), order='C')
    cdef FLOAT[:] sines = np.pad(np.sin(theta), (0, 1)), cosines = np.pad(np.cos(theta), (0, 1))
    cdef FLOAT angle_weight = np.pi / (2 * n_angles)
    cdef Py_ssize_t tiles_per_side = (size + TILE - 1) // TILE
    cdef Py_ssize_t tile, top, left, bottom, right, first, second, row, col, s
    cdef FLOAT first_cos, first_sin, second_cos, second_sin, first_across, second_across
    cdef const FLOAT* first_projection
    cdef const FLOAT* second_projection
    cdef FLOAT* tile_sums

    for tile in prange(tiles_per_side * tiles_per_side, nogil=True, num_threads=num_threads):
        top = tile // tiles_per_side * TILE
        left = tile % tiles_per_side * TILE
        bottom = min(top + TILE, size)
        right = min(left + TILE, size)
        tile_sums = &sums[threadid(), 0]
        memset(tile_sums, 0, TILE * TILE * n_slices * sizeof(FLOAT))

        for first in range(0, n_angles, 2):
            second = first + 1
            first_projection = &projections[first, 0, 0]
            second_projection = &projections[second, 0, 0] if second < n_angles else &zeros[0, 0]
            first_cos, first_sin = cosines[first], sines[first]
            second_cos, second_sin = cosines[second], sines[second]

            for row in range(top, bottom):
                first_across = -xs[row] * first_sin
                second_across = -xs[row] * second_sin

                for col in range(left, right):
                    if inside_circle[row, col]:
                        add_two_angles(
                            pixel_sums(tile_sums, row - top, col - left, n_slices),
                            first_projection, xs[col] * first_cos + first_across + centre,
                            second_projection, xs[col] * second_cos + second_across + centre,
                            n_slices,
                        )

        for s in range(n_slices):
            for row in range(top, bottom):
                for col in range(left, right):
                    if inside_circle[row, col]:
                        image[s, row, col] = pixel_sums(tile_sums, row - top, col - left, n_slices)[s] * angle_weight
                    else:
                        image[s, row, col] = fill_value

    return np.asarray(image)
