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
from libc.math cimport ceil, floor

from ._utils cimport FP32_FP64, get_pixel2d


# most of this code is taken from skimage and optimized for direct Radon transform
# TODO: move interpolateNd-s to _utils.pyx
cdef inline FP32_FP64 interpolate2d(FP32_FP64* input,
                                    Py_ssize_t rows, Py_ssize_t cols,
                                    double r, double c,
                                    double cval) nogil:
    cdef double dr, dc
    cdef long minr, minc, maxr, maxc

    minr = <long>floor(r)
    minc = <long>floor(c)
    maxr = minr + 1
    maxc = minc + 1

    dr = r - minr
    dc = c - minc

    cdef double c0, c1

    cdef double c00 = get_pixel2d(input, rows, cols, minr, minc, cval)
    cdef double c01 = get_pixel2d(input, rows, cols, minr, maxc, cval)
    cdef double c10 = get_pixel2d(input, rows, cols, maxr, minc, cval)
    cdef double c11 = get_pixel2d(input, rows, cols, maxr, maxc, cval)

    c0 = c00 * (1 - dc) + c01 * dc
    c1 = c10 * (1 - dc) + c11 * dc

    return c0 * (1 - dr) + c1 * dr


cdef inline FP32_FP64 accumulate(FP32_FP64* image, Py_ssize_t* size, FP32_FP64* sin, FP32_FP64* cos,
                                 FP32_FP64* r_shift, FP32_FP64* c_shift, Py_ssize_t j, Py_ssize_t* limit) nogil:
    cdef FP32_FP64 result = 0
    cdef Py_ssize_t i

    for i in range(limit[0], size[0] - limit[0]):
        result += interpolate2d(
            image, size[0], size[0],
            j * (-sin[0]) + i * cos[0] - c_shift[0],
            j * cos[0] + i * sin[0] - r_shift[0],
            0,
        )

    return result


def radon3d(FP32_FP64[:, :, :] image, FP32_FP64[:] theta, Py_ssize_t[:] limits, Py_ssize_t num_threads):
    cdef Py_ssize_t size = image.shape[1], n_slices = image.shape[0], n_angles = len(theta)
    cdef FP32_FP64[:, :, ::1] img = np.ascontiguousarray(image)
    cdef FP32_FP64[:, :, :] out = np.zeros_like(img, shape=(n_slices, size, n_angles))

    sins = np.sin(theta)
    coss = np.cos(theta)
    cdef FP32_FP64 center = size // 2
    cdef FP32_FP64[:] sinuses = sins
    cdef FP32_FP64[:] cosinuses = coss
    cdef FP32_FP64[:] r_shift = center * (coss + sins - 1)
    cdef FP32_FP64[:] c_shift = center * (coss - sins - 1)

    cdef Py_ssize_t i, j, alpha, slc
    cdef FP32_FP64 r, c

    for slc in prange(n_slices, nogil=True, num_threads=num_threads):
        for alpha in prange(n_angles):
            for j in prange(size):
                out[slc, j, alpha] = accumulate(
                    &img[slc, 0, 0], &size,
                    &sinuses[alpha], &cosinuses[alpha], &r_shift[alpha], &c_shift[alpha],
                    j, &limits[j],
                )

    return np.asarray(out)
