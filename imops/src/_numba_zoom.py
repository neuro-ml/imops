import numpy as np
from numba import njit, prange


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
def adjusted_coef(old_n: int, new_n: int) -> float:
    if new_n == 1:
        return old_n
    return (np.float64(old_n) - 1) / (np.float64(new_n) - 1)


@njit(nogil=True)
def get_pixel3d(input: np.ndarray, rows: int, cols: int, dims: int, r: int, c: int, d: int, cval: float) -> float:
    if 0 <= r < rows and 0 <= c < cols and 0 <= d < dims:
        return input[r, c, d]

    return cval


@njit(nogil=True)
def interpolate3d(input: np.ndarray, rows: int, cols: int, dims: int, r: int, c: int, d: int, cval: float) -> float:
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


def _zoom(input: np.ndarray, zoom: np.ndarray, cval: float) -> np.ndarray:
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
                zoomed[i, j, k] = interpolate3d(
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
