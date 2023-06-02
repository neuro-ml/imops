from typing import Union

import numpy as np
from numba import njit, prange


float_or_int = Union[float, int]


def _interp1d(
    input: np.ndarray,
    old_locations: np.ndarray,
    new_locations: np.ndarray,
    bounds_error: bool,
    fill_value: float,
    extrapolate: bool,
    assume_sorted: bool,
) -> np.ndarray:
    rows, cols, dims = input.shape[0], input.shape[1], len(new_locations)
    contiguous_input = np.ascontiguousarray(input)

    dtype = input.dtype
    interpolated = np.zeros((rows, cols, dims), dtype=dtype)
    dd = np.zeros(dims)

    old_dims = len(old_locations)
    sort_permutation = np.arange(old_dims) if assume_sorted else np.argsort(old_locations)
    max_idxs = np.searchsorted(old_locations[sort_permutation], new_locations)

    extr = np.zeros(dims, dtype=np.int8)

    for k in prange(dims):
        if max_idxs[k] == 0:
            if new_locations[k] < old_locations[sort_permutation[max_idxs[k]]]:
                extr[k] = -1
            else:
                max_idxs[k] = 1

        if max_idxs[k] >= old_dims:
            extr[k] = 1

        if extr[k] == 0:
            dd[k] = (new_locations[k] - old_locations[sort_permutation[max_idxs[k] - 1]]) / (
                old_locations[sort_permutation[max_idxs[k]]] - old_locations[sort_permutation[max_idxs[k] - 1]]
            )

    if bounds_error and np.any(extr):
        raise ValueError('A value in x_new is out of the interpolation range.')

    if np.any(extr) and extrapolate:
        slope_left = np.zeros((rows, cols))
        slope_right = np.zeros((rows, cols))
        bias_left = np.zeros((rows, cols))
        bias_right = np.zeros((rows, cols))

        slope_left = get_slope(
            old_locations[sort_permutation[0]],
            contiguous_input[..., sort_permutation[0]],
            old_locations[sort_permutation[1]],
            contiguous_input[..., sort_permutation[1]],
        )
        slope_right = get_slope(
            old_locations[sort_permutation[old_dims - 1]],
            contiguous_input[..., sort_permutation[old_dims - 1]],
            old_locations[sort_permutation[old_dims - 2]],
            contiguous_input[..., sort_permutation[old_dims - 2]],
        )

        bias_left = contiguous_input[..., sort_permutation[0]] - slope_left * old_locations[sort_permutation[0]]
        bias_right = (
            contiguous_input[..., sort_permutation[old_dims - 1]]
            - slope_right * old_locations[sort_permutation[old_dims - 1]]
        )

    for i in prange(rows):
        for j in prange(cols):
            for k in prange(dims):
                if extr[k] == 0:
                    interpolated[i, j, k] = (
                        contiguous_input[i, j, sort_permutation[max_idxs[k] - 1]] * (1 - dd[k])
                        + contiguous_input[i, j, sort_permutation[max_idxs[k]]] * dd[k]
                    )
                elif extrapolate:
                    if extr[k] == 1:
                        interpolated[i, j, k] = slope_right[i, j] * new_locations[k] + bias_right[i, j]
                    else:
                        interpolated[i, j, k] = slope_left[i, j] * new_locations[k] + bias_left[i, j]
                else:
                    interpolated[i, j, k] = fill_value

    return interpolated


@njit(nogil=True)
def get_slope(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return (y2 - y1) / (x2 - x1)


@njit(nogil=True)
def get_pixel3d(
    input: np.ndarray, rows: int, cols: int, dims: int, r: int, c: int, d: int, cval: float_or_int
) -> float_or_int:
    if 0 <= r < rows and 0 <= c < cols and 0 <= d < dims:
        return input[r, c, d]

    return cval


@njit(nogil=True)
def get_pixel4d(
    input: np.ndarray,
    dim1: int,
    dim2: int,
    dim3: int,
    dim4: int,
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    cval: float_or_int,
) -> float_or_int:
    if 0 <= c1 < dim1 and 0 <= c2 < dim2 and 0 <= c3 < dim3 and 0 <= c4 < dim4:
        return input[c1, c2, c3, c4]

    return cval


@njit(nogil=True)
def adjusted_coef(old_n: int, new_n: int) -> float:
    if new_n == 1:
        return old_n
    return (np.float64(old_n) - 1) / (np.float64(new_n) - 1)


@njit(nogil=True)
def distance3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


@njit(nogil=True)
def distance4d(x1: float, y1: float, z1: float, d1: float, x2: float, y2: float, z2: float, d2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2 + (d1 - d2) ** 2) ** 0.5


@njit(nogil=True)
def interpolate3d_linear(
    input: np.ndarray, rows: int, cols: int, dims: int, r: float, c: float, d: float, cval: float
) -> float:
    minr = int(r)
    minc = int(c)
    mind = int(d)
    maxr = minr + 1
    maxc = minc + 1
    maxd = mind + 1

    dr = r - minr
    dc = c - minc
    dd = d - mind

    c000 = get_pixel3d(input, rows, cols, dims, minr, minc, mind, cval)
    c001 = get_pixel3d(input, rows, cols, dims, minr, minc, maxd, cval)
    c010 = get_pixel3d(input, rows, cols, dims, minr, maxc, mind, cval)
    c011 = get_pixel3d(input, rows, cols, dims, minr, maxc, maxd, cval)
    c100 = get_pixel3d(input, rows, cols, dims, maxr, minc, mind, cval)
    c101 = get_pixel3d(input, rows, cols, dims, maxr, minc, maxd, cval)
    c110 = get_pixel3d(input, rows, cols, dims, maxr, maxc, mind, cval)
    c111 = get_pixel3d(input, rows, cols, dims, maxr, maxc, maxd, cval)

    c00 = c000 * (1 - dr) + c100 * dr
    c01 = c001 * (1 - dr) + c101 * dr
    c10 = c010 * (1 - dr) + c110 * dr
    c11 = c011 * (1 - dr) + c111 * dr

    c0 = c00 * (1 - dc) + c10 * dc
    c1 = c01 * (1 - dc) + c11 * dc

    return c0 * (1 - dd) + c1 * dd


@njit(nogil=True)
def interpolate3d_nearest(
    input: np.ndarray, rows: int, cols: int, dims: int, r: float, c: float, d: float, cval: float_or_int
) -> float_or_int:
    min_distance = 3.0
    i_nearest, j_nearest, k_nearest = -1, -1, -1

    minr = int(r)
    minc = int(c)
    mind = int(d)
    maxr = minr + 1
    maxc = minc + 1
    maxd = mind + 1

    for i in range(2):
        curr = maxr if i else minr
        if curr >= rows:
            continue
        for j in range(2):
            curc = maxc if j else minc
            if curc >= cols:
                continue
            for k in range(2):
                curd = maxd if k else mind
                if curd >= dims:
                    continue

                distance = distance3d(r, c, d, curr, curc, curd)

                if distance <= min_distance:
                    i_nearest = i
                    j_nearest = j
                    k_nearest = k
                    min_distance = distance

    if i_nearest == -1 or j_nearest == -1 or k_nearest == -1:
        return cval

    return get_pixel3d(
        input,
        rows,
        cols,
        dims,
        maxr if i_nearest else minr,
        maxc if j_nearest else minc,
        maxd if k_nearest else mind,
        cval,
    )


@njit(nogil=True)
def interpolate4d_linear(
    input: np.ndarray,
    dim1: int,
    dim2: int,
    dim3: int,
    dim4: int,
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    cval: float,
) -> float:
    minc1 = int(c1)
    minc2 = int(c2)
    minc3 = int(c3)
    minc4 = int(c4)
    maxc1 = minc1 + 1
    maxc2 = minc2 + 1
    maxc3 = minc3 + 1
    maxc4 = minc4 + 1

    dc1 = c1 - minc1
    dc2 = c2 - minc2
    dc3 = c3 - minc3
    dc4 = c4 - minc4

    c0000 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, minc2, minc3, minc4, cval)
    c0001 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, minc2, minc3, maxc4, cval)
    c0010 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, minc2, maxc3, minc4, cval)
    c0011 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, minc2, maxc3, maxc4, cval)
    c0100 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, maxc2, minc3, minc4, cval)
    c0101 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, maxc2, minc3, maxc4, cval)
    c0110 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, maxc2, maxc3, minc4, cval)
    c0111 = get_pixel4d(input, dim1, dim2, dim3, dim4, minc1, maxc2, maxc3, maxc4, cval)
    c1000 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, minc2, minc3, minc4, cval)
    c1001 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, minc2, minc3, maxc4, cval)
    c1010 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, minc2, maxc3, minc4, cval)
    c1011 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, minc2, maxc3, maxc4, cval)
    c1100 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, maxc2, minc3, minc4, cval)
    c1101 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, maxc2, minc3, maxc4, cval)
    c1110 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, maxc2, maxc3, minc4, cval)
    c1111 = get_pixel4d(input, dim1, dim2, dim3, dim4, maxc1, maxc2, maxc3, maxc4, cval)

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


@njit(nogil=True)
def interpolate4d_nearest(
    input: np.ndarray,
    dim1: int,
    dim2: int,
    dim3: int,
    dim4: int,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    cval: float_or_int,
) -> float_or_int:
    min_distance = 3.0
    i1_nearest, i2_nearest, i3_nearest, i4_nearest = -1, -1, -1, -1
    minc1 = int(c1)
    minc2 = int(c2)
    minc3 = int(c3)
    minc4 = int(c4)
    maxc1 = minc1 + 1
    maxc2 = minc2 + 1
    maxc3 = minc3 + 1
    maxc4 = minc4 + 1

    for i1 in range(2):
        curc1 = maxc1 if i1 else minc1
        if curc1 >= dim1:
            continue
        for i2 in range(2):
            curc2 = maxc2 if i2 else minc2
            if curc2 >= dim2:
                continue
            for i3 in range(2):
                curc3 = maxc3 if i3 else minc3
                if curc3 >= dim3:
                    continue
                for i4 in range(2):
                    curc4 = maxc4 if i4 else minc4
                    if curc4 >= dim4:
                        continue

                    distance = distance4d(c1, c2, c3, c4, curc1, curc2, curc3, curc4)
                    if distance <= min_distance:
                        i1_nearest = i1
                        i2_nearest = i2
                        i3_nearest = i3
                        i4_nearest = i4
                        min_distance = distance

    if i1_nearest == -1 or i2_nearest == -1 or i3_nearest == -1 or i4_nearest == -1:
        return cval

    return get_pixel4d(
        input,
        dim1,
        dim2,
        dim3,
        dim4,
        maxc1 if i1_nearest else minc1,
        maxc2 if i2_nearest else minc2,
        maxc3 if i3_nearest else minc3,
        maxc4 if i4_nearest else minc4,
        cval,
    )


def _zoom3d_linear(input: np.ndarray, zoom: np.ndarray, cval: float) -> np.ndarray:
    contiguous_input = np.ascontiguousarray(input)

    old_rows, old_cols, old_dims = input.shape
    row_coef, col_coef, dim_coef = zoom

    new_shape = (round(old_rows * row_coef), round(old_cols * col_coef), round(old_dims * dim_coef))
    new_rows, new_cols, new_dims = new_shape

    zoomed = np.zeros(new_shape, dtype=input.dtype)

    adjusted_row_coef = adjusted_coef(old_rows, new_rows)
    adjusted_col_coef = adjusted_coef(old_cols, new_cols)
    adjusted_dim_coef = adjusted_coef(old_dims, new_dims)

    for i in prange(new_rows):
        for j in prange(new_cols):
            for k in prange(new_dims):
                zoomed[i, j, k] = interpolate3d_linear(
                    contiguous_input,
                    old_rows,
                    old_cols,
                    old_dims,
                    i * adjusted_row_coef,
                    j * adjusted_col_coef,
                    k * adjusted_dim_coef,
                    cval,
                )

    return zoomed


def _zoom3d_nearest(input: np.ndarray, zoom: np.ndarray, cval: float_or_int) -> np.ndarray:
    contiguous_input = np.ascontiguousarray(input)

    old_rows, old_cols, old_dims = input.shape
    row_coef, col_coef, dim_coef = zoom

    new_shape = (round(old_rows * row_coef), round(old_cols * col_coef), round(old_dims * dim_coef))
    new_rows, new_cols, new_dims = new_shape

    zoomed = np.zeros(new_shape, dtype=input.dtype)

    adjusted_row_coef = adjusted_coef(old_rows, new_rows)
    adjusted_col_coef = adjusted_coef(old_cols, new_cols)
    adjusted_dim_coef = adjusted_coef(old_dims, new_dims)

    for i in prange(new_rows):
        for j in prange(new_cols):
            for k in prange(new_dims):
                zoomed[i, j, k] = interpolate3d_nearest(
                    contiguous_input,
                    old_rows,
                    old_cols,
                    old_dims,
                    i * adjusted_row_coef,
                    j * adjusted_col_coef,
                    k * adjusted_dim_coef,
                    cval,
                )

    return zoomed


def _zoom4d_linear(input: np.ndarray, zoom: np.ndarray, cval: float) -> np.ndarray:
    contiguous_input = np.ascontiguousarray(input)

    old_dim1, old_dim2, old_dim3, old_dim4 = input.shape
    dim1_coef, dim2_coef, dim3_coef, dim4_coef = zoom

    new_shape = (
        round(old_dim1 * dim1_coef),
        round(old_dim2 * dim2_coef),
        round(old_dim3 * dim3_coef),
        round(old_dim4 * dim4_coef),
    )
    new_dim1, new_dim2, new_dim3, new_dim4 = new_shape

    zoomed = np.zeros(new_shape, dtype=input.dtype)

    adjusted_dim1_coef = adjusted_coef(old_dim1, new_dim1)
    adjusted_dim2_coef = adjusted_coef(old_dim2, new_dim2)
    adjusted_dim3_coef = adjusted_coef(old_dim3, new_dim3)
    adjusted_dim4_coef = adjusted_coef(old_dim4, new_dim4)

    for i1 in prange(new_dim1):
        for i2 in prange(new_dim2):
            for i3 in prange(new_dim3):
                for i4 in prange(new_dim4):
                    zoomed[i1, i2, i3, i4] = interpolate4d_linear(
                        contiguous_input,
                        old_dim1,
                        old_dim2,
                        old_dim3,
                        old_dim4,
                        i1 * adjusted_dim1_coef,
                        i2 * adjusted_dim2_coef,
                        i3 * adjusted_dim3_coef,
                        i4 * adjusted_dim4_coef,
                        cval,
                    )

    return zoomed


def _zoom4d_nearest(input: np.ndarray, zoom: np.ndarray, cval: float_or_int) -> np.ndarray:
    contiguous_input = np.ascontiguousarray(input)

    old_dim1, old_dim2, old_dim3, old_dim4 = input.shape
    dim1_coef, dim2_coef, dim3_coef, dim4_coef = zoom

    new_shape = (
        round(old_dim1 * dim1_coef),
        round(old_dim2 * dim2_coef),
        round(old_dim3 * dim3_coef),
        round(old_dim4 * dim4_coef),
    )
    new_dim1, new_dim2, new_dim3, new_dim4 = new_shape

    zoomed = np.zeros(new_shape, dtype=input.dtype)

    adjusted_dim1_coef = adjusted_coef(old_dim1, new_dim1)
    adjusted_dim2_coef = adjusted_coef(old_dim2, new_dim2)
    adjusted_dim3_coef = adjusted_coef(old_dim3, new_dim3)
    adjusted_dim4_coef = adjusted_coef(old_dim4, new_dim4)

    for i1 in prange(new_dim1):
        for i2 in prange(new_dim2):
            for i3 in prange(new_dim3):
                for i4 in prange(new_dim4):
                    zoomed[i1, i2, i3, i4] = interpolate4d_nearest(
                        contiguous_input,
                        old_dim1,
                        old_dim2,
                        old_dim3,
                        old_dim4,
                        i1 * adjusted_dim1_coef,
                        i2 * adjusted_dim2_coef,
                        i3 * adjusted_dim3_coef,
                        i4 * adjusted_dim4_coef,
                        cval,
                    )

    return zoomed
