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
        intBounds[i].assigned = False

    prev_x = vx[nr_verts - 1]
    prev_y = vy[nr_verts - 1]

    for i in range(nr_verts):
        curr_x = vx[i]
        curr_y = vy[i]

        if prev_x == curr_x:
            x_set = <int>prev_x
            fpBounds[x_set] = set_bound(set_bound(fpBounds[x_set], prev_y), curr_y)
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

            x_start = <int>ceilf(tmp_from_x)
            x_stop = <int>ceilf(tmp_to_x)

            for x_set in range(x_start, x_stop):
                lerp_t = (x_set - tmp_from_x) / (tmp_to_x - tmp_from_x)
                bound_y = lerp(tmp_from_y, tmp_to_y, lerp_t)
                fpBounds[x_set] = set_bound(fpBounds[x_set], bound_y)

        prev_x = curr_x
        prev_y = curr_y

    for m in range(M):
        intBounds[m] = intify(fpBounds[m], 0, N)

    for m in range(M):
        tmp_int_bound = intBounds[m]

        if tmp_int_bound.assigned:
            for n in range(tmp_int_bound.lb, tmp_int_bound.rb):
                out[m, n] = True


    return out


cdef inline intBound intify(fpBound bound, Py_ssize_t min_idx, Py_ssize_t max_idx):
    if bound.assigned:
        return intBound(lb = max(min_idx, <int>floorf(bound.lb)), rb = min(max_idx, <int>ceilf(bound.rb)), assigned=True)

    return intBound(lb=0, rb=0, assigned=False)


cdef inline fpBound set_bound(fpBound bound, float new_bound):
    cdef new_lb, new_rb

    if bound.assigned:
        new_lb = min(bound.lb, new_bound)
        new_rb = max(bound.rb, new_bound)
    else:
        new_lb = new_bound
        new_rb = new_bound

    return fpBound(lb=new_lb, rb=new_rb, assigned=True)


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
