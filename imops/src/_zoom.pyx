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


cdef inline FLOAT get_pixel4d(FLOAT* input,
                              Py_ssize_t stride1, Py_ssize_t stride2, Py_ssize_t stride3, Py_ssize_t stride4,
                              Py_ssize_t dim1, Py_ssize_t dim2, Py_ssize_t dim3, Py_ssize_t dim4,
                              Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3, Py_ssize_t c4,
                              FLOAT cval) nogil:
        if (c1 < 0) or (c1 >= dim1) or (c2 < 0) or (c2 >= dim2) or (c3 < 0) or (c3 >= dim3) or (c4 < 0) or (c4 >= dim4):
            return cval
        return input[c1 * stride1 + c2 * stride2 + c3 * stride3 + c4 * stride4]


cdef inline double adjusted_coef(Py_ssize_t old_n, Py_ssize_t new_n) nogil:
    if new_n == 1:
        return old_n
    return  (<double>old_n - 1) / (<double>new_n - 1)


cdef inline FLOAT interpolate3d(FLOAT* input,
                                 Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t dims,
                                 double r, double c, double d,
                                 double cval) nogil:
    cdef double dr, dc, dd
    cdef long minr, minc, mind, maxr, maxc, maxd

    minr = <long>floor(r)
    minc = <long>floor(c)
    mind = <long>floor(d)
    maxr = minr + 1
    maxc = minc + 1
    maxd = mind + 1

    dr = r - minr
    dc = c - minc
    dd = d - mind

    cdef double c00, c01, c10, c11, c0, c1

    cdef double c000 = get_pixel3d(input, rows, cols, dims, minr, minc, mind, cval)
    cdef double c001 = get_pixel3d(input, rows, cols, dims, minr, minc, maxd, cval)
    cdef double c010 = get_pixel3d(input, rows, cols, dims, minr, maxc, mind, cval)
    cdef double c011 = get_pixel3d(input, rows, cols, dims, minr, maxc, maxd, cval)
    cdef double c100 = get_pixel3d(input, rows, cols, dims, maxr, minc, mind, cval)
    cdef double c101 = get_pixel3d(input, rows, cols, dims, maxr, minc, maxd, cval)
    cdef double c110 = get_pixel3d(input, rows, cols, dims, maxr, maxc, mind, cval)
    cdef double c111 = get_pixel3d(input, rows, cols, dims, maxr, maxc, maxd, cval)

    c00 = c000 * (1 - dr) + c100 * dr
    c01 = c001 * (1 - dr) + c101 * dr
    c10 = c010 * (1 - dr) + c110 * dr
    c11 = c011 * (1 - dr) + c111 * dr

    c0 = c00 * (1 - dc) + c10 * dc
    c1 = c01 * (1 - dc) + c11 * dc

    return c0 * (1 - dd) + c1 * dd


cdef inline FLOAT interpolate4d(FLOAT* input,
                                Py_ssize_t stride1, Py_ssize_t stride2, Py_ssize_t stride3, Py_ssize_t stride4,
                                Py_ssize_t dim1, Py_ssize_t dim2, Py_ssize_t dim3, Py_ssize_t dim4,
                                double c1, double c2, double c3, double c4,
                                double cval) nogil:
    cdef double dc1, dc2, dc3, dc4
    cdef long minc1, minc2, minc3, minc4, maxc1, maxc2, maxc3, maxc4

    minc1 = <long>floor(c1)
    minc2 = <long>floor(c2)
    minc3 = <long>floor(c3)
    minc4 = <long>floor(c4)
    maxc1 = minc1 + 1
    maxc2 = minc2 + 1
    maxc3 = minc3 + 1
    maxc4 = minc4 + 1

    dc1 = c1 - minc1
    dc2 = c2 - minc2
    dc3 = c3 - minc3
    dc4 = c4 - minc4

    cdef double c000, c001, c010, c011, c100, c101, c110, c111, c00, c01, c10, c11, c0_, c1_

    cdef double c0000 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, minc2, minc3, minc4, cval)
    cdef double c0001 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, minc2, minc3, maxc4, cval)
    cdef double c0010 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, minc2, maxc3, minc4, cval)
    cdef double c0011 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, minc2, maxc3, maxc4, cval)
    cdef double c0100 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, maxc2, minc3, minc4, cval)
    cdef double c0101 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, maxc2, minc3, maxc4, cval)
    cdef double c0110 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, maxc2, maxc3, minc4, cval)
    cdef double c0111 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, minc1, maxc2, maxc3, maxc4, cval)
    cdef double c1000 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, minc2, minc3, minc4, cval)
    cdef double c1001 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, minc2, minc3, maxc4, cval)
    cdef double c1010 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, minc2, maxc3, minc4, cval)
    cdef double c1011 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, minc2, maxc3, maxc4, cval)
    cdef double c1100 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, maxc2, minc3, minc4, cval)
    cdef double c1101 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, maxc2, minc3, maxc4, cval)
    cdef double c1110 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, maxc2, maxc3, minc4, cval)
    cdef double c1111 = get_pixel4d(input, stride1, stride2, stride3, stride4, dim1, dim2, dim3, dim4, maxc1, maxc2, maxc3, maxc4, cval)

    c000 = c0000 * (1 - dc1) + c1000 * dc1
    c001 = c0001 * (1 - dc1) + c1001 * dc1
    c010 = c0010 * (1 - dc1) + c1010 * dc1
    c011 = c0011 * (1 - dc1) + c1011 * dc1
    c100 = c0100 * (1 - dc1) + c1100 * dc1
    c101 = c0101 * (1 - dc1) + c1101 * dc1
    c110 = c0110 * (1 - dc1) + c1110 * dc1
    c111 = c0111 * (1 - dc1) + c1111 * dc1

    c00 = c000 * (1 - dc2) + c100 * dc2
    c01 = c001 * (1 - dc2) + c101 * dc2
    c10 = c010 * (1 - dc2) + c110 * dc2
    c11 = c011 * (1 - dc2) + c111 * dc2

    c0_ = c00 * (1 - dc3) + c10 * dc3
    c1_ = c01 * (1 - dc3) + c11 * dc3

    return c0_ * (1 - dc4) + c1_ * dc4


def _zoom3d(FLOAT[:, :, :] input, double[:] zoom, double cval, Py_ssize_t num_threads):
    cdef FLOAT[:, :, ::1] contiguous_input = np.ascontiguousarray(input)

    cdef Py_ssize_t old_rows = input.shape[0], old_cols = input.shape[1], old_dims = input.shape[2]
    cdef double row_coef = zoom[0], col_coef = zoom[1], dim_coef = zoom[2]

    new_shape = (round(old_rows * row_coef), round(old_cols * col_coef), round(old_dims * dim_coef))
    cdef Py_ssize_t new_rows = new_shape[0], new_cols = new_shape[1], new_dims = new_shape[2]

    cdef FLOAT[:, :, ::1] zoomed = np.zeros(new_shape, dtype=np.float32 if input.itemsize == 4 else np.float64)

    cdef Py_ssize_t i, j, k
    cdef double adjusted_row_coef, adjusted_col_coef, adjusted_dim_coef

    adjusted_row_coef = adjusted_coef(old_rows, new_rows)
    adjusted_col_coef = adjusted_coef(old_cols, new_cols)
    adjusted_dim_coef = adjusted_coef(old_dims, new_dims)

    for i in prange(new_rows, nogil=True, num_threads=num_threads):
        for j in prange(new_cols):
            for k in prange(new_dims):
                zoomed[i, j, k] = interpolate3d(
                    &contiguous_input[0, 0, 0],
                    old_rows, old_cols, old_dims,
                    i * adjusted_row_coef, j * adjusted_col_coef, k * adjusted_dim_coef, cval,
                )

    return np.asarray(zoomed)


def _zoom4d(FLOAT[:, :, :, :] input, double[:] zoom, double cval, Py_ssize_t num_threads):
    cdef FLOAT[:, :, :, ::1] contiguous_input = np.ascontiguousarray(input)

    cdef Py_ssize_t old_dim1 = input.shape[0], old_dim2 = input.shape[1], old_dim3 = input.shape[2], old_dim4 = input.shape[3]
    cdef double dim1_coef = zoom[0], dim2_coef = zoom[1], dim3_coef = zoom[2], dim4_coef = zoom[3]
    new_shape = (
        round(old_dim1 * dim1_coef),
        round(old_dim2 * dim2_coef),
        round(old_dim3 * dim3_coef),
        round(old_dim4 * dim4_coef),
    )
    cdef Py_ssize_t new_dim1 = new_shape[0], new_dim2 = new_shape[1], new_dim3 = new_shape[2], new_dim4 = new_shape[3]

    cdef FLOAT[:, :, :, ::1] zoomed = np.zeros(new_shape, dtype=np.float32 if input.itemsize == 4 else np.float64)

    cdef Py_ssize_t i1, i2, i3, i4, stride1, stride2, stride3, stride4
    cdef double adjusted_dim1_coef, adjusted_dim2_coef, adjusted_dim3_coef, adjusted_dim4_coef

    adjusted_dim1_coef = adjusted_coef(old_dim1, new_dim1)
    adjusted_dim2_coef = adjusted_coef(old_dim2, new_dim2)
    adjusted_dim3_coef = adjusted_coef(old_dim3, new_dim3)
    adjusted_dim4_coef = adjusted_coef(old_dim4, new_dim4)

    stride1 = old_dim2 * old_dim3 * old_dim4
    stride2 = old_dim3 * old_dim4
    stride3 = old_dim4
    stride4 = 1

    for i1 in prange(new_dim1, nogil=True, num_threads=num_threads):
        for i2 in prange(new_dim2):
            for i3 in prange(new_dim3):
                for i4 in prange(new_dim4):
                    zoomed[i1, i2, i3, i4] = interpolate4d(
                        &contiguous_input[0, 0, 0, 0],
                        stride1, stride2, stride3, stride4,
                        old_dim1, old_dim2, old_dim3, old_dim4,
                        i1 * adjusted_dim1_coef,
                        i2 * adjusted_dim2_coef,
                        i3 * adjusted_dim3_coef,
                        i4 * adjusted_dim4_coef,
                        cval,
                    )

    return np.asarray(zoomed)
