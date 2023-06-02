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
from libc.math cimport floor


ctypedef cython.floating FLOAT
ctypedef np.uint8_t uint8


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT interpolate(FLOAT x, FLOAT* ys, FLOAT radius, FLOAT right_limit) nogil:
    cdef Py_ssize_t idx
    cdef FLOAT val, value

    value = x + radius
    if value < 0 or value > right_limit:
        val = 0

    else:
        idx = <int>floor(value)

        if idx == right_limit:
            val = ys[idx]
        else:
            val = (ys[idx + 1] - ys[idx]) * (value - idx) + ys[idx]

    return val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef FLOAT accumulate(FLOAT x, FLOAT y, FLOAT* sinuses, FLOAT* cosinuses, FLOAT* ys,
                      Py_ssize_t size, Py_ssize_t image_size, FLOAT radius, FLOAT right_limit) nogil:
    cdef FLOAT accumulator = 0
    cdef Py_ssize_t k

    for k in range(0, size):
        accumulator += interpolate(y * cosinuses[k] - x * sinuses[k], ys + k * image_size, radius, right_limit)
    return accumulator


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FLOAT[:, :, :] backprojection3d(FLOAT[:, :, :] sinogram, FLOAT[:] theta, FLOAT[:] xs,
                                      uint8[:, :] inside_circle, FLOAT fill_value, int image_size, int output_size,
                                      Py_ssize_t num_threads):
    cdef FLOAT[:, :, :] result = np.zeros_like(sinogram, shape=(len(sinogram), output_size, output_size))
    cdef Py_ssize_t slc, i, j, n_angles = len(theta), n_slices = len(sinogram)
    cdef FLOAT min_val = image_size // 2, right_lim = image_size - 1
    cdef FLOAT[:] sinuses = np.sin(theta)
    cdef FLOAT[:] cosinuses = np.cos(theta)
    cdef FLOAT multiplier = np.pi / (2 * n_angles)

    sinogram = np.ascontiguousarray(np.moveaxis(sinogram, -1, -2))

    for slc in prange(0, n_slices, nogil=True, num_threads=num_threads):
        for i in prange(0, output_size):
            for j in prange(0, output_size):
                if inside_circle[i, j]:
                    result[slc, i, j] = accumulate(
                        xs[i], xs[j],
                        &sinuses[0], &cosinuses[0], &sinogram[slc, 0, 0],
                        n_angles, image_size, min_val, right_lim
                    ) * multiplier
                else:
                    result[slc, i, j] = fill_value

    return np.asarray(result)
