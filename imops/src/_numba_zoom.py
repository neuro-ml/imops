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
def get_pixel5d(
    input: np.ndarray,
    dim1: int,
    dim2: int,
    dim3: int,
    dim4: int,
    dim5: int,
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    c5: int,
    cval: float,
) -> float:
    if 0 <= c1 < dim1 and 0 <= c2 < dim2 and 0 <= c3 < dim3 and 0 <= c4 < dim4 and 0 <= c5 < dim5:
        return input[c1, c2, c3, c4, c5]

    return cval


@njit(nogil=True)
def interpolate5d(
    input: np.ndarray,
    dim1: int,
    dim2: int,
    dim3: int,
    dim4: int,
    dim5: int,
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    c5: int,
    cval: float,
) -> float:
    minc1 = int(c1)
    minc2 = int(c2)
    minc3 = int(c3)
    minc4 = int(c4)
    minc5 = int(c5)
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

    c00000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, minc4, minc5, cval)
    c00001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, minc4, maxc5, cval)
    c00010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, maxc4, minc5, cval)
    c00011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, minc3, maxc4, maxc5, cval)
    c00100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, minc4, minc5, cval)
    c00101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, minc4, maxc5, cval)
    c00110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, maxc4, minc5, cval)
    c00111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, minc2, maxc3, maxc4, maxc5, cval)
    c01000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, minc4, minc5, cval)
    c01001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, minc4, maxc5, cval)
    c01010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, maxc4, minc5, cval)
    c01011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, minc3, maxc4, maxc5, cval)
    c01100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, minc4, minc5, cval)
    c01101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, minc4, maxc5, cval)
    c01110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, maxc4, minc5, cval)
    c01111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, minc1, maxc2, maxc3, maxc4, maxc5, cval)
    c10000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, minc4, minc5, cval)
    c10001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, minc4, maxc5, cval)
    c10010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, maxc4, minc5, cval)
    c10011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, minc3, maxc4, maxc5, cval)
    c10100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, minc4, minc5, cval)
    c10101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, minc4, maxc5, cval)
    c10110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, maxc4, minc5, cval)
    c10111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, minc2, maxc3, maxc4, maxc5, cval)
    c11000 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, minc4, minc5, cval)
    c11001 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, minc4, maxc5, cval)
    c11010 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, maxc4, minc5, cval)
    c11011 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, minc3, maxc4, maxc5, cval)
    c11100 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, minc4, minc5, cval)
    c11101 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, minc4, maxc5, cval)
    c11110 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, maxc4, minc5, cval)
    c11111 = get_pixel5d(input, dim1, dim2, dim3, dim4, dim5, maxc1, maxc2, maxc3, maxc4, maxc5, cval)

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


def _zoom(input: np.ndarray, zoom: np.ndarray, cval: float) -> np.ndarray:
    contiguous_input = np.ascontiguousarray(input)

    old_dim1, old_dim2, old_dim3, old_dim4, old_dim5 = input.shape
    dim1_coef, dim2_coef, dim3_coef, dim4_coef, dim5_coef = zoom

    new_shape = (
        round(old_dim1 * dim1_coef),
        round(old_dim2 * dim2_coef),
        round(old_dim3 * dim3_coef),
        round(old_dim4 * dim4_coef),
        round(old_dim5 * dim5_coef),
    )
    new_dim1, new_dim2, new_dim3, new_dim4, new_dim5 = new_shape

    zoomed = np.zeros(new_shape, dtype=input.dtype)

    adjusted_dim1_coef = adjusted_coef(old_dim1, new_dim1)
    adjusted_dim2_coef = adjusted_coef(old_dim2, new_dim2)
    adjusted_dim3_coef = adjusted_coef(old_dim3, new_dim3)
    adjusted_dim4_coef = adjusted_coef(old_dim4, new_dim4)
    adjusted_dim5_coef = adjusted_coef(old_dim5, new_dim5)

    for i1 in prange(new_dim1):
        for i2 in prange(new_dim2):
            for i3 in prange(new_dim3):
                for i4 in prange(new_dim4):
                    for i5 in prange(new_dim5):
                        zoomed[i1, i2, i3, i4, i5] = interpolate5d(
                            contiguous_input,
                            old_dim1,
                            old_dim2,
                            old_dim3,
                            old_dim4,
                            old_dim5,
                            i1 * adjusted_dim1_coef,
                            i2 * adjusted_dim2_coef,
                            i3 * adjusted_dim3_coef,
                            i4 * adjusted_dim4_coef,
                            i5 * adjusted_dim5_coef,
                            cval,
                        )

    return zoomed
