# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

import numpy as np

cimport numpy as np

from cython.parallel import prange

from libc.stdlib cimport abort, free, malloc


cdef struct array_iterator:
    int rank_m1
    int dimensions[3]
    int coordinates[3]
    int strides[3]
    int backstrides[3]


cdef int init_array_iterator(array_iterator *arr_iter, int *shape, int *strides) nogil:
    cdef int i

    arr_iter.rank_m1 = 2

    for i in range(3):
        arr_iter[0].dimensions[i] = shape[i] - 1
        arr_iter[0].coordinates[i] = 0
        arr_iter[0].strides[i] = strides[i]
        arr_iter[0].backstrides[i] = strides[i] * arr_iter[0].dimensions[i]

    return 0


cdef struct filter_iterator:
    int strides[3]
    int backstrides[3]
    int bound1[3]
    int bound2[3]


cdef int init_filter_iterator(
    filter_iterator *filter_iter,
    int *a_shape,
    int *f_shape,
    int filter_size,
    np.uint8_t is_dilation
) nogil:
    cdef int i, rank = 3, step, orgn

    filter_iter[0].strides[rank - 1] = filter_size
    for i in range(rank - 2, -1, -1):
        step = a_shape[i + 1] if a_shape[i + 1] < f_shape[i + 1] else f_shape[i + 1]
        filter_iter[0].strides[i] = filter_iter[0].strides[i + 1] * step

    for i in range(rank):
        step = a_shape[i] if a_shape[i] < f_shape[i] else f_shape[i]
        orgn = f_shape[i] // 2
        if is_dilation and f_shape[i] % 2 == 0:
            orgn -= 1

        filter_iter[0].backstrides[i] = (step - 1) * filter_iter[0].strides[i]
        filter_iter[0].bound1[i] = orgn
        filter_iter[0].bound2[i] = a_shape[i] - f_shape[i] + orgn

    return 0


cdef int init_filter_offsets(
    int *a_shape, int *a_strides,
    np.uint8_t *footprint, int *f_shape,
    int footprint_size,
    int **offsets, int *border_flag_value,
    np.uint8_t is_dilation
) nogil:
    cdef int i, j, k
    cdef int filter_size = 1, offsets_size = 1
    cdef int max_size = 0, max_stride = 0
    cdef int coordinates[3]
    cdef int position[3]
    cdef int *po
    cdef int stride, offset, orgn, cc

    for i in range(3):
        filter_size *= f_shape[i]

    for i in range(3):
        offsets_size *= (a_shape[i] if a_shape[i] < f_shape[i] else f_shape[i])

    offsets[0] = <int*>malloc(offsets_size * footprint_size * sizeof(int))
    if offsets[0] == NULL:
        abort()

    for i in range(3):
        if a_shape[i] > max_size:
            max_size = a_shape[i]

        stride = -a_strides[i] if a_strides[i] < 0 else a_strides[i]

        if stride > max_stride:
            max_stride = stride

        coordinates[i] = 0
        position[i] = 0

    border_flag_value[0] = max_size * max_stride + 1
    po = offsets[0]

    for j in range(offsets_size):
        for k in range(filter_size):
            offset = 0

            # only calculate an offset if the footprint is 1:
            if footprint[k]:
                # find offsets along all axes:
                for i in range(3):
                    orgn = f_shape[i] // 2
                    if is_dilation and f_shape[i] % 2 == 0:
                        orgn -= 1
                    cc = coordinates[i] - orgn + position[i]

                    if cc < 0 or cc >= a_shape[i]:
                        offset = border_flag_value[0]
                        break
                    else:
                        # use an offset that is possibly mapped from outside the border:
                        cc -= position[i]
                        offset += a_strides[i] * cc

                # store the offset
                po[0] = offset
                po += 1

            # next point in the filter:
            for i in range(3 - 1, -1, -1):
                if coordinates[i] < f_shape[i] - 1:
                    coordinates[i] += 1
                    break
                else:
                    coordinates[i] = 0

        # move to the next array region:
        for i in range(3 - 1, -1, -1):
            orgn = f_shape[i] // 2
            if is_dilation and f_shape[i] % 2 == 0:
                orgn -= 1

            if position[i] == orgn:
                position[i] += a_shape[i] - f_shape[i] + 1

                if position[i] <= orgn:
                    position[i] = orgn + 1
            else:
                position[i] += 1

            if position[i] < a_shape[i]:
                break
            else:
                position[i] = 0

    return 0


cdef inline int filter_iterator_offset(filter_iterator *filter_iter, int *coordinates) nogil:
    cdef int i, position, offset = 0

    for i in range(3):
        if filter_iter[0].bound1[i] > filter_iter[0].bound2[i]:
            position = coordinates[i]
        elif coordinates[i] < filter_iter[0].bound1[i]:
            position = coordinates[i]
        elif coordinates[i] >= filter_iter[0].bound2[i]:
            position = filter_iter[0].bound1[i] + coordinates[i] - filter_iter[0].bound2[i]
        else:
            position = filter_iter[0].bound1[i]

        offset += filter_iter[0].strides[i] * position

    return offset


cdef inline int worker(
    np.uint8_t *input, np.uint8_t *footprint, np.uint8_t *output,
    int *a_shape, int *a_strides, int *f_shape,
    int footprint_size,
    int *offsets, int border_flag_value,
    int start, int end,
    np.uint8_t border_value,
    np.uint8_t is_dilation,
) nogil:
    cdef np.uint8_t _true = not is_dilation
    cdef np.uint8_t _false = not _true

    cdef array_iterator input_iter
    cdef filter_iterator filter_iter

    init_array_iterator(&input_iter, a_shape, a_strides)

    init_filter_iterator(&filter_iter, a_shape, f_shape, footprint_size, is_dilation)

    cdef int temp = start
    for i in range(3):
        input_iter.coordinates[i] = temp // input_iter.strides[i]
        temp %= input_iter.strides[i]

    cdef np.uint8_t *pi = input + start
    cdef np.uint8_t *po = output + start
    cdef int *oo = offsets + filter_iterator_offset(&filter_iter, <int*>input_iter.coordinates)

    cdef np.uint8_t out
    cdef int _oo, _pp

    for j in range(start, end):
        out = _true

        for i in range(footprint_size):
            _oo = oo[i]

            if _oo == border_flag_value:
                if border_value == _false:
                    out = _false
                    break
            elif pi[_oo] == _false:
                out = _false
                break

        po[0] = out

        # pointers and filter next
        for i in range(input_iter.rank_m1, -1, -1):
            _pp = input_iter.coordinates[i]

            if _pp < input_iter.dimensions[i]:
                if _pp < filter_iter.bound1[i] or _pp >= filter_iter.bound2[i]:
                    oo += filter_iter.strides[i]

                input_iter.coordinates[i] += 1
                break
            else:
                input_iter.coordinates[i] = 0

                oo -= filter_iter.backstrides[i]

        pi += 1
        po += 1

    return 0


def _binary_operation(
    np.uint8_t[:, :, :] input,
    np.uint8_t[:, :, :] footprint,
    np.uint8_t[:, :, ::1] out,
    Py_ssize_t num_threads,
    np.uint8_t border_value,
    np.uint8_t is_dilation,
) -> np.ndarray:
    cdef np.uint8_t[:, :, ::1] c_input = np.ascontiguousarray(input)
    cdef np.uint8_t[:, :, ::1] c_footprint = np.ascontiguousarray(footprint)

    cdef int f_shape[3]
    cdef int a_shape[3]
    cdef int a_strides[3]
    for i in range(3):
        f_shape[i] = c_footprint.shape[i]
        a_shape[i] = c_input.shape[i]
        a_strides[i] = c_input.strides[i]

    cdef int footprint_size = np.sum(c_footprint)
    cdef int size = np.size(c_input)

    cdef int *offsets
    cdef int border_flag_value

    init_filter_offsets(
        a_shape, a_strides,
        &c_footprint[0, 0, 0], f_shape, footprint_size,
        &offsets, &border_flag_value,
        is_dilation
    )

    cdef int task, start, chunk

    cdef int _mod = size % num_threads, _div = size // num_threads

    for task in prange(num_threads, nogil=True, num_threads=num_threads):
        if task < _mod:
            chunk = _div + 1
            start = chunk * task
        else:
            chunk = _div
            start = chunk * task + _mod

        worker(
            &c_input[0, 0, 0], &c_footprint[0, 0, 0], &out[0, 0, 0],
            a_shape, a_strides, f_shape,
            footprint_size,
            offsets, border_flag_value,
            start, start + chunk,
            border_value,
            is_dilation
        )

    free(offsets)

    return np.asarray(out)


def _binary_erosion(
    np.uint8_t[:, :, :] input,
    np.uint8_t[:, :, :] footprint,
    np.uint8_t[:, :, ::1] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    return _binary_operation(input, footprint, out, num_threads, True, False)


def _binary_dilation(
    np.uint8_t[:, :, :] input,
    np.uint8_t[:, :, :] footprint,
    np.uint8_t[:, :, ::1] out,
    Py_ssize_t num_threads,
) -> np.ndarray:
    inverted_footprint = np.array(footprint[::-1, ::-1, ::-1])
    return _binary_operation(input, inverted_footprint, out, num_threads, False, True)
