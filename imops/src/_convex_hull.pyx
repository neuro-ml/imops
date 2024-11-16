#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

from libc.math cimport ceilf, floorf

cimport numpy as cnp
cimport cython

cnp.import_array()

FP_BOUND_DTYPE = np.dtype([
    ('lb', np.float32),
    ('rb', np.float32),
    ('assigned', np.uint8)
])

INT_BOUND_DTYPE = np.dtype([
    ('lb', np.int32),
    ('rb', np.int32),
    ('assigned', np.uint8)
])


def _grid_points_in_poly(float[:] vx, float[:] vy, Py_ssize_t M, Py_ssize_t N, Py_ssize_t nr_verts):
    cdef intBound tmp_int_bound
    cdef Py_ssize_t m, n, i
    cdef float prev_x, prev_y, curr_x, curr_ys
    cdef float tmp_from_x, tmp_from_y, tmp_to_x, tmp_to_y
    cdef float lerp_t, bound_y
    cdef Py_ssize_t x_set, x_start, x_stop

    cdef cnp.ndarray[dtype=cnp.uint8_t, ndim=2, mode="c"] out = \
         np.zeros((M, N), dtype=np.uint8)

    cdef fpBound[:] fpBounds = np.empty(M, dtype=FP_BOUND_DTYPE)
    cdef intBound[:] intBounds = np.empty(M, dtype=INT_BOUND_DTYPE)

    for i in range(M):
        fpBounds[i].assigned = False
        fpBounds[i].lb = float('inf')
        fpBounds[i].rb = -1
        intBounds[i].assigned = False

    prev_x = vx[nr_verts - 1]
    prev_y = vy[nr_verts - 1]

    # algorithm relies on vertex validity and counterclockwise orientation of the vertices
    for i in range(nr_verts):
        curr_x = vx[i]
        curr_y = vy[i]

        if prev_x == curr_x:
            x_set = <int>(floorf(prev_x) if prev_y < curr_y else ceilf(prev_x))

            fpBounds[x_set].assigned = True
            fpBounds[x_set].lb = min(fpBounds[x_set].lb, prev_y, curr_y)
            fpBounds[x_set].rb = max(fpBounds[x_set].rb, prev_y, curr_y)
        else:
            if prev_x < curr_x:
                tmp_from_x = prev_x
                tmp_from_y = prev_y
                tmp_to_x = curr_x
                tmp_to_y = curr_y
            else:
                tmp_from_x = curr_x
                tmp_from_y = curr_y
                tmp_to_x = prev_x
                tmp_to_y = prev_y

            # vertices are treated as points on image, so include x_stop
            x_start = <int>ceilf(tmp_from_x)
            x_stop = <int>floorf(tmp_to_x + 1)

            for x_set in range(x_start, x_stop):
                lerp_t = (x_set - tmp_from_x) / (tmp_to_x - tmp_from_x)
                bound_y = lerp(tmp_from_y, tmp_to_y, lerp_t)

                fpBounds[x_set].assigned = True
                fpBounds[x_set].lb = min(fpBounds[x_set].lb, bound_y)
                fpBounds[x_set].rb = max(fpBounds[x_set].rb, bound_y)

        prev_x = curr_x
        prev_y = curr_y

    # bounds are computed as point interpolation
    # so bounds must be valid indices for out array 
    for m in range(M):
        intBounds[m] = intify(fpBounds[m], 0, N - 1)

    for m in range(M):
        tmp_int_bound = intBounds[m]

        if tmp_int_bound.assigned:
            # Do not forget to fill right bound
            for n in range(tmp_int_bound.lb, tmp_int_bound.rb + 1):
                out[m, n] = True

    return out


cdef inline intBound intify(fpBound bound, Py_ssize_t min_idx, Py_ssize_t max_idx):
    if bound.assigned:
        return intBound(lb = max(min_idx, <int>floorf(bound.lb)), rb = min(max_idx, <int>ceilf(bound.rb)), assigned=True)

    return intBound(lb=0, rb=0, assigned=False)


cdef packed struct fpBound:
    float lb
    float rb
    unsigned char assigned


cdef packed struct intBound:
    int lb
    int rb
    unsigned char assigned


cdef inline float lerp(float y0, float y1, float t):
    return y0 * (1 - t) + y1 * t


cpdef _left_right_bounds(cnp.uint8_t[:,:] image):
    cdef Py_ssize_t i, j, M = image.shape[0], N = image.shape[1], curr_pos = 0, left, right
    cdef cnp.ndarray[dtype=int, ndim=2, mode="c"] left_right_bounds = np.zeros((2 * M, 2), dtype=np.int32)
    cdef unsigned char found = False

    for i in range(M):
        found = False

        for j in range(N):
            if image[i, j]:
                left = j
                found = True
                break

        for j in range(N):
            if image[i, N - 1 - j]:
                right = N - 1 - j
                found = True
                break

        if found:
            left_right_bounds[2 * curr_pos, 0] = i
            left_right_bounds[2 * curr_pos, 1] = left
            left_right_bounds[2 * curr_pos + 1, 0] = i
            left_right_bounds[2 * curr_pos + 1, 1] = right

            curr_pos += 1

    return np.ascontiguousarray(left_right_bounds[: 2 * curr_pos, :])
