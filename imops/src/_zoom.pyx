# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange

from libc.math cimport floor


ctypedef cython.floating FLOAT


def _interp1d(FLOAT[:, :, :] input,
              double[:] old_locations, double[:] new_locations,
              np.uint8_t bounds_error, double fill_value, np.uint8_t extrapolate, np.uint8_t assume_sorted,
              Py_ssize_t num_threads) -> np.ndarray:
    cdef Py_ssize_t rows = input.shape[0], cols = input.shape[1], dims = len(new_locations)
    cdef FLOAT[:, :, ::1] contiguous_input = np.ascontiguousarray(input)

    dtype = np.float32 if input.itemsize == 4 else np.float64
    cdef FLOAT[:, :, ::1] interpolated = np.zeros((rows, cols, dims), dtype=dtype)
    cdef double[:] dd = np.zeros(dims)

    cdef Py_ssize_t old_dims = len(old_locations)
    cdef Py_ssize_t[:] sort_permutation = np.arange(old_dims) if assume_sorted else np.argsort(old_locations)
    cdef long[:] max_idxs = np.searchsorted(old_locations, new_locations, sorter=sort_permutation)

    cdef Py_ssize_t i, j, k
    cdef char[:] extr = np.zeros(dims, dtype=np.int8)
    cdef double[:, ::1] slope_left, slope_right, bias_left, bias_right

    for k in prange(dims, nogil=True, num_threads=num_threads):
        if max_idxs[k] == 0:
            if new_locations[k] < old_locations[sort_permutation[max_idxs[k]]]:
                extr[k] = -1
            else:
                max_idxs[k] = 1

        if max_idxs[k] >= old_dims:
            extr[k] = 1

        if extr[k] == 0:
            dd[k] = ((new_locations[k] - old_locations[sort_permutation[max_idxs[k] - 1]]) /
                     (old_locations[sort_permutation[max_idxs[k]]] - old_locations[sort_permutation[max_idxs[k] - 1]]))

    if bounds_error and np.any(extr):
        raise ValueError('A value in x_new is out of the interpolation range.')

    if np.any(extr) and extrapolate:
        slope_left = np.zeros((rows, cols))
        slope_right = np.zeros((rows, cols))
        bias_left = np.zeros((rows, cols))
        bias_right = np.zeros((rows, cols))

        for i in prange(rows, nogil=True, num_threads=num_threads):
            for j in prange(cols):
                slope_left[i, j] = get_slope(
                    old_locations[sort_permutation[0]],
                    contiguous_input[i, j, sort_permutation[0]],
                    old_locations[sort_permutation[1]],
                    contiguous_input[i, j, sort_permutation[1]],
                )
                slope_right[i, j] = get_slope(
                    old_locations[sort_permutation[old_dims - 1]],
                    contiguous_input[i, j, sort_permutation[old_dims - 1]],
                    old_locations[sort_permutation[old_dims - 2]],
                    contiguous_input[i, j, sort_permutation[old_dims - 2]],
                )

                bias_left[i, j] = contiguous_input[i, j, sort_permutation[0]] - slope_left[i, j] * old_locations[sort_permutation[0]]
                bias_right[i, j] = contiguous_input[i, j, sort_permutation[old_dims - 1]] - slope_right[i, j] * old_locations[sort_permutation[old_dims - 1]]

    for i in prange(rows, nogil=True, num_threads=num_threads):
        for j in prange(cols):
            for k in prange(dims):
                if extr[k] == 0:
                    interpolated[i, j, k] = (
                        get_pixel3d(
                            &contiguous_input[0, 0, 0],
                            rows, cols, old_dims,
                            i, j, sort_permutation[max_idxs[k] - 1],
                            0,
                        )  * (1 - dd[k]) +
                        get_pixel3d(
                            &contiguous_input[0, 0, 0],
                            rows, cols, old_dims,
                            i, j, sort_permutation[max_idxs[k]],
                            0,
                        ) * dd[k])
                elif extrapolate:
                    if extr[k] == 1:
                        interpolated[i, j, k] = slope_right[i, j] * new_locations[k] + bias_right[i, j]
                    else:
                        interpolated[i, j, k] = slope_left[i, j] * new_locations[k] + bias_left[i, j]
                else:
                    interpolated[i, j, k] = fill_value

    return np.asarray(interpolated)


cdef inline double get_slope(double x1, double y1, double x2, double y2) nogil:
    return (y2 - y1) / (x2 - x1)


cdef inline FLOAT get_pixel5d(FLOAT* input,
                              Py_ssize_t dim1, Py_ssize_t dim2, Py_ssize_t dim3, Py_ssize_t dim4, Py_ssize_t dim5,
                              Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5,
                              FLOAT cval, layout = b'C') nogil:
        if (c1 < 0) or (c1 >= dim1) or (c2 < 0) or (c2 >= dim2) or (c3 < 0) or (c3 >= dim3) or (c4 < 0) or (c4 >= dim4) or (c5 < 0) or (c5 >= dim5):
            return cval

        if layout == b'C':
            return input[c1 * dim2 * dim3 * dim4 * dim5 + c2 * dim3 * dim4 * dim5 + c3 * dim4 * dim5 + c4 * dim5 + c5]

        if layout == b'F':
            return input[dim1 * dim2 * dim3 * dim4 * c5 + dim1 * dim2 * dim3 * c4 + dim1 * dim2 * c3 + dim1 * c2 + c1]


cdef inline FLOAT get_pixel3d(FLOAT* input,
                              Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                              Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                              FLOAT cval, layout = b'C') nogil:
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols) or (d < 0) or (d >= dims):
            return cval

        if layout == b'C':
            return input[r * cols * dims + c * dims + d]

        if layout == b'F':
            return input[rows * cols * d + rows * c + r]


cdef inline double adjusted_coef(Py_ssize_t old_n, Py_ssize_t new_n) nogil:
    if new_n == 1:
        return old_n
    return  (<double>old_n - 1) / (<double>new_n - 1)


cdef inline FLOAT interpolate5d(FLOAT* input,
                                Py_ssize_t dim1, Py_ssize_t dim2, Py_ssize_t dim3, Py_ssize_t dim4, Py_ssize_t dim5,
                                double c1, double c2, double c3, double c4, double c5,
                                double cval) nogil:
    cdef double dc1, dc2, dc3, dc4, dc5
    cdef long minc1, minc2, minc3, minc4, minc5, maxc1, maxc2, maxc3, maxc4, maxc5

    minc1 = <long>floor(c1)
    minc2 = <long>floor(c2)
    minc3 = <long>floor(c3)
    minc4 = <long>floor(c4)
    minc5 = <long>floor(c5)
    maxc1 = minc1 + 1
    maxc2 = minc2 + 1
    maxc3 = minc3 + 1
    maxc4 = minc4 + 1
    maxc5 = minc5 + 1

    dc1 = c1 - minc1
    dc2 = c2 - minc2
    dc3 = c3 - minc3
    dc4 = c4 - minc4
    dc5 = c5 - minc5

    cdef double c0000, c0001, c0010, c0011, c0100, c0101, c0110, c0111, c1000, c1001, c1010, c1011, c1100, c1101, c1110, c1111
    cdef double c000, c001, c010, c011, c100, c101, c110, c111, c00, c01, c10, c11, c0_, c1_

    cdef double c00000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, minc4, minc5, cval)
    cdef double c00001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, minc4, maxc5, cval)
    cdef double c00010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, maxc4, minc5, cval)
    cdef double c00011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, maxc4, maxc5, cval)
    cdef double c00100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, minc4, minc5, cval)
    cdef double c00101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, minc4, maxc5, cval)
    cdef double c00110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, maxc4, minc5, cval)
    cdef double c00111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, maxc4, maxc5, cval)
    cdef double c01000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, minc4, minc5, cval)
    cdef double c01001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, minc4, maxc5, cval)
    cdef double c01010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, maxc4, minc5, cval)
    cdef double c01011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, maxc4, maxc5, cval)
    cdef double c01100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, minc4, minc5, cval)
    cdef double c01101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, minc4, maxc5, cval)
    cdef double c01110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, maxc4, minc5, cval)
    cdef double c01111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, maxc4, maxc5, cval)
    cdef double c10000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, minc4, minc5, cval)
    cdef double c10001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, minc4, maxc5, cval)
    cdef double c10010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, maxc4, minc5, cval)
    cdef double c10011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, maxc4, maxc5, cval)
    cdef double c10100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, minc4, minc5, cval)
    cdef double c10101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, minc4, maxc5, cval)
    cdef double c10110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, maxc4, minc5, cval)
    cdef double c10111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, maxc4, maxc5, cval)
    cdef double c11000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, minc4, minc5, cval)
    cdef double c11001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, minc4, maxc5, cval)
    cdef double c11010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, maxc4, minc5, cval)
    cdef double c11011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, maxc4, maxc5, cval)
    cdef double c11100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, minc4, minc5, cval)
    cdef double c11101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, minc4, maxc5, cval)
    cdef double c11110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, maxc4, minc5, cval)
    cdef double c11111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, maxc4, maxc5, cval)

    c0000 = c00000 * (1 - dc1) + c10000 * dc1
    c0001 = c00001 * (1 - dc1) + c10001 * dc1
    c0010 = c00010 * (1 - dc1) + c10010 * dc1
    c0011 = c00011 * (1 - dc1) + c10011 * dc1
    c0100 = c00100 * (1 - dc1) + c10100 * dc1
    c0101 = c00101 * (1 - dc1) + c10101 * dc1
    c0110 = c00110 * (1 - dc1) + c10110 * dc1
    c0111 = c00111 * (1 - dc1) + c10111 * dc1
    c1000 = c01000 * (1 - dc1) + c11000 * dc1
    c1001 = c01001 * (1 - dc1) + c11001 * dc1
    c1010 = c01010 * (1 - dc1) + c11010 * dc1
    c1011 = c01011 * (1 - dc1) + c11011 * dc1
    c1100 = c01100 * (1 - dc1) + c11100 * dc1
    c1101 = c01101 * (1 - dc1) + c11101 * dc1
    c1110 = c01110 * (1 - dc1) + c11110 * dc1
    c1111 = c01111 * (1 - dc1) + c11111 * dc1

    c000 = c0000 * (1 - dc2) + c1000 * dc2
    c001 = c0001 * (1 - dc2) + c1001 * dc2
    c010 = c0010 * (1 - dc2) + c1010 * dc2
    c011 = c0011 * (1 - dc2) + c1011 * dc2
    c100 = c0100 * (1 - dc2) + c1100 * dc2
    c101 = c0101 * (1 - dc2) + c1101 * dc2
    c110 = c0110 * (1 - dc2) + c1110 * dc2
    c111 = c0111 * (1 - dc2) + c1111 * dc2

    c00 = c000 * (1 - dc3) + c100 * dc3
    c01 = c001 * (1 - dc3) + c101 * dc3
    c10 = c010 * (1 - dc3) + c110 * dc3
    c11 = c011 * (1 - dc3) + c111 * dc3

    c0_ = c00 * (1 - dc4) + c10 * dc4
    c1_ = c01 * (1 - dc4) + c11 * dc4

    return c0_ * (1 - dc5) + c1_ * dc5


def _zoom(FLOAT[:, :, :, :, :] input, double[:] zoom, double cval, Py_ssize_t num_threads):
    cdef FLOAT[:, :, :, :, ::1] contiguous_input = np.ascontiguousarray(input)

    cdef Py_ssize_t old_dim1 = input.shape[0], old_dim2 = input.shape[1], old_dim3 = input.shape[2], old_dim4 = input.shape[3], old_dim5 = input.shape[4]
    cdef double dim1_coef = zoom[0], dim2_coef = zoom[1], dim3_coef = zoom[2], dim4_coef = zoom[3], dim5_coef = zoom[4]
    new_shape = (
        round(old_dim1 * dim1_coef),
        round(old_dim2 * dim2_coef),
        round(old_dim3 * dim3_coef),
        round(old_dim4 * dim4_coef),
        round(old_dim5 * dim5_coef),
    )
    cdef Py_ssize_t new_dim1 = new_shape[0], new_dim2 = new_shape[1], new_dim3 = new_shape[2], new_dim4 = new_shape[3], new_dim5 = new_shape[4]

    cdef FLOAT[:, :, :, :, ::1] zoomed = np.zeros(new_shape, dtype=np.float32 if input.itemsize == 4 else np.float64)

    cdef Py_ssize_t i1, i2, i3, i4, i5
    cdef double adjusted_dim1_coef, adjusted_dim2_coef, adjusted_dim3_coef, adjusted_dim4_coef, adjusted_dim5_coef

    adjusted_dim1_coef = adjusted_coef(old_dim1, new_dim1)
    adjusted_dim2_coef = adjusted_coef(old_dim2, new_dim2)
    adjusted_dim3_coef = adjusted_coef(old_dim3, new_dim3)
    adjusted_dim4_coef = adjusted_coef(old_dim4, new_dim4)
    adjusted_dim5_coef = adjusted_coef(old_dim5, new_dim5)

    for i1 in prange(new_dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(new_dim2):
            for i3 in prange(new_dim3):
                for i4 in prange(new_dim4):
                    for i5 in prange(new_dim5):
                        zoomed[i1, i2, i3, i4, i5] = interpolate5d(
                            &contiguous_input[0, 0, 0, 0, 0],
                            old_dim1, old_dim2, old_dim3, old_dim4, old_dim5,
                            i1 * adjusted_dim1_coef,
                            i2 * adjusted_dim2_coef,
                            i3 * adjusted_dim3_coef,
                            i4 * adjusted_dim4_coef,
                            i5 * adjusted_dim5_coef,
                            cval,
                        )

    return np.asarray(zoomed)
