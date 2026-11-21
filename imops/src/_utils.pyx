# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=True
# cython: overflowcheck.fold=False
# cython: embedsignature=False
# cython: embedsignature.format=c
# cython: c_api_binop_methods=True
# cython: profile=False
# cython: linetrace=False
# cython: infer_types=False
# cython: language_level=3
# cython: cpp_locals=False


from libcpp.unordered_set cimport unordered_set

import numpy as np

cimport numpy as np

from cython.parallel import prange


ctypedef fused NUM:
    short
    int
    long long


cpdef void _isin(const NUM[:] elements, const NUM[:] test_elements, np.uint8_t[:] res, Py_ssize_t num_threads):
    cdef unordered_set[NUM] test_elements_set
    cdef Py_ssize_t i
    cdef Py_ssize_t elements_len = elements.shape[0]
    cdef Py_ssize_t test_elements_len = test_elements.shape[0]

    test_elements_set.reserve(test_elements_len)

    with nogil:
        for i in range(test_elements_len):
            test_elements_set.insert(test_elements[i])

        for i in prange(elements_len, num_threads=num_threads):
            res[i] = test_elements_set.count(elements[i])


cdef extern from *:
    """
    #if defined(_MSC_VER) && defined(_M_X64)
    #include <intrin.h>
    static int cpu_supports_x86_64_v3(void) {
        /* FMA MOVBE OSXSAVE AVX F16C */
        const int leaf1_ecx = (1 << 12) | (1 << 22) | (1 << 27) | (1 << 28) | (1 << 29);
        /* BMI1 AVX2 BMI2 */
        const int leaf7_ebx = (1 << 3) | (1 << 5) | (1 << 8);
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return 0;
        __cpuid(r, 1);
        if ((r[2] & leaf1_ecx) != leaf1_ecx) return 0;
        if ((_xgetbv(0) & 6) != 6) return 0;  /* the OS saves YMM registers */
        __cpuidex(r, 7, 0);
        if ((r[1] & leaf7_ebx) != leaf7_ebx) return 0;
        __cpuid(r, 0x80000001);
        return (r[2] >> 5) & 1;  /* LZCNT */
    }
    #elif defined(__x86_64__)
    static int cpu_supports_x86_64_v3(void) {
        __builtin_cpu_init();
        /* GCC 11 has no "x86-64-v3" name, so the features are listed */
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("bmi")
            && __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("movbe") && __builtin_cpu_supports("f16c")
            && __builtin_cpu_supports("lzcnt");
    }
    #else
    static int cpu_supports_x86_64_v3(void) { return 0; }
    #endif
    """
    bint cpu_supports_x86_64_v3()


def supports_x86_64_v3():
    return cpu_supports_x86_64_v3()
