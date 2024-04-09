#include <x86intrin.h>
#include <iostream>
#include <vector>
// #include <boost/align.hpp>
// #include <stdlib.h>

// template <typename T, size_t Alignment = 1>
// using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, Alignment>>;

int main() {
    int16_t* p = reinterpret_cast<int16_t*>(aligned_alloc(64, 64));
    for (size_t i = 0; i < 32; ++i) {
        p[i] = 2 * static_cast<int16_t>(i);
    }
    for (size_t i = 0; i < 32; ++i) {
        std::cout << p[i] << ' ';
    }
    std::cout << std::endl;
    __m512i mm = _mm512_load_si512(p);
    int16_t* p_mm = reinterpret_cast<int16_t*>(&mm[0]);
    for (size_t i = 0; i < 32; ++i) {
        std::cout << p_mm[i] << ' ';
    }
    free(p);
    std::cout << std::endl;
}