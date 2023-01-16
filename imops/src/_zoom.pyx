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


def _zoom(FLOAT[:, :, :] input, double[:] zoom, double cval, Py_ssize_t num_threads):
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
