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


ctypedef cython.floating FLOAT
ctypedef cython.integral INT

# most of this code is taken from skimage and optimized for direct Radon transform

cdef inline FLOAT get_pixel2d(FLOAT* image, Py_ssize_t size, long r, long c) nogil:
    if (r < 0) or (r >= size) or (c < 0) or (c >= size):
        return 0
    else:
        return image[r * size + c]


cdef inline FLOAT interpolate2d(FLOAT* image, Py_ssize_t size, FLOAT r, FLOAT c) nogil:
    cdef FLOAT dr, dc
    cdef long minr, minc, maxr, maxc

    minr = <long>floor(r)
    minc = <long>floor(c)
    maxr = <long>ceil(r)
    maxc = <long>ceil(c)
    dr = r - minr
    dc = c - minc

    cdef FLOAT top
    cdef FLOAT bottom

    cdef FLOAT top_left = get_pixel2d(image, size, minr, minc)
    cdef FLOAT top_right = get_pixel2d(image, size, minr, maxc)
    cdef FLOAT bottom_left = get_pixel2d(image, size, maxr, minc)
    cdef FLOAT bottom_right = get_pixel2d(image, size, maxr, maxc)

    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right
    return (1 - dr) * top + dr * bottom


cdef inline FLOAT accumulate(FLOAT* image, Py_ssize_t* size, FLOAT* sin, FLOAT* cos,
                             FLOAT* r_shift, FLOAT* c_shift, Py_ssize_t j, INT* limit) nogil:
    cdef FLOAT result = 0
    cdef Py_ssize_t i

    for i in range(limit[0], size[0] - limit[0]):
        result += interpolate2d(
            image, size[0],
            j * (-sin[0]) + i * cos[0] - c_shift[0],
            j * cos[0] + i * sin[0] - r_shift[0],
        )

    return result


def radon3d(FLOAT[:, :, :] image, FLOAT[:] theta, INT[:] limits, Py_ssize_t num_threads):
    cdef Py_ssize_t size = image.shape[1], n_slices = image.shape[0], n_angles = len(theta)
    cdef FLOAT[:, :, ::1] img = np.ascontiguousarray(image)
    cdef FLOAT[:, :, :] out = np.zeros_like(img, shape=(n_slices, size, n_angles))

    sins = np.sin(theta)
    coss = np.cos(theta)
    cdef FLOAT center = size // 2
    cdef FLOAT[:] sinuses = sins
    cdef FLOAT[:] cosinuses = coss
    cdef FLOAT[:] r_shift = center * (coss + sins - 1)
    cdef FLOAT[:] c_shift = center * (coss - sins - 1)

    cdef Py_ssize_t i, alpha, slc

    for slc in prange(n_slices, nogil=True, num_threads=num_threads):
        for alpha in prange(n_angles):
            for i in prange(size):
                out[slc, i, alpha] = accumulate(
                    &img[slc, 0, 0], &size,
                    &sinuses[alpha], &cosinuses[alpha], &r_shift[alpha], &c_shift[alpha],
                    i, &limits[i],
                )

    return np.asarray(out)
