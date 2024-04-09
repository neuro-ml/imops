#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <boost/align.hpp>
#include <x86intrin.h>

namespace get_contour_segments_stacked {

namespace py = pybind11;
using pyarr_float = py::array_t<float>;
template <typename T, size_t Alignment = 1>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, Alignment>>;

inline float get_fraction(float from_value, float to_value, float level) {
    return to_value == from_value ? 0 : ((level - from_value) / (to_value - from_value));
}

size_t get_contour_segments_stacked_func(pyarr_float& array_input, float level) {
    std::vector<py::list> output;
    auto array = array_input.unchecked<3>();
    int dim_Z = array.shape(0);
    int dim_R = array.shape(1);
    int dim_C = array.shape(2);

    // constexpr alignas(64) int8_t ones[64];
    // for (size_t i = 0; i < 64; ++i) {
    //     ones[i] = static_cast<int8_t>(1);
    // }
    // constexpr __m512i mm_ones = _mm512_load_si512(ones);
    constexpr __m512i mm_ones = _mm512_set1_epi8(1);
    constexpr __m512i mm_neg_ones = _mm512_set1_epi8(-1);

    aligned_vector<int8_t, 64> vec(4 * array.size());
    // float* p = static_cast<float*>(array_input.request().ptr);
    // for (size_t i = 0; i < vec.size(); ++i) {
    //     [[likely]] vec[i] = std::isnan(p[i]) ? static_cast<int8_t>(-1)
    //                               : static_cast<int8_t>(p[i] != static_cast<float>(0));
    // }
    size_t flatten_index = 0;
    for (int z = 0; z < dim_Z; ++z) {
        for (int r = 0; r < dim_R - 1; ++r) {
            for (int c = 0; c < dim_C - 1; ++c) {
                int r1 = r + 1;
                int c1 = c + 1;
                vec[flatten_index++] = static_cast<int8_t>(array(z, r, c));
                vec[flatten_index++] = static_cast<int8_t>(array(z, r, c1));
                vec[flatten_index++] = static_cast<int8_t>(array(z, r1, c));
                vec[flatten_index++] = static_cast<int8_t>(array(z, r1, c1));
            }
        }
    }

    int sZ = array_input.strides(0);
    int sR = array_input.strides(1);
    int sC = array_input.strides(2);

    for (size_t i = 0; i < vec.size(); i += 64) {
        __m512i elems = _mm512_load_si512(&vec[i]);
        __m512i diff = _mm512_subs_epi8(mm_ones, elems);

    }

    for (int z_index = 0; z_index < dim_Z; ++z_index) {
        int z_flatten = sR * sC * z_index;
        std::vector<py::tuple> segments;
        for (int r0 = 0; r0 < dim_R - 1; ++r0) {
            for (int c0 = 0; c0 < dim_C - 1; ++c0) {

                int r0_flatten = z_flatten + sC * r0;
                int r1_flatten = r0_flatten + sC;

                int r1 = r0 + 1;
                int c1 = c0 + 1;

                float ul = static_cast<float>(vec[z_flatten + r0_flatten + c0]);
                float ur = static_cast<float>(vec[z_flatten + r0_flatten + c1]);
                float ll = static_cast<float>(vec[z_flatten + r1_flatten + c0]);
                float lr = static_cast<float>(vec[z_flatten + r1_flatten + c1]);

                // float ul = array(z_index, r0, c0);
                // float ur = array(z_index, r0, c1);
                // float ll = array(z_index, r1, c0);
                // float lr = array(z_index, r1, c1);

                // if (std::isnan(ul) or std::isnan(ur) or std::isnan(ll) or std::isnan(lr)) {
                //     continue;
                // }
                if (ul == -1 or ur == -1 or ll == -1 or lr == -1) {
                    continue;
                }

                size_t square_case = 0;
                if (ul > level) {
                    ++square_case;
                }
                if (ur > level) {
                    square_case += 2;
                }
                if (ll > level) {
                    square_case += 4;
                }
                if (lr > level) {
                    square_case += 8;
                }
                if (square_case == 0 or square_case == 15) {
                    continue;
                }

                py::tuple top = py::make_tuple(r0, c0 + get_fraction(ul, ur, level));
                py::tuple bottom = py::make_tuple(r1, c0 + get_fraction(ll, lr, level));
                py::tuple left = py::make_tuple(r0 + get_fraction(ul, ll, level), c0);
                py::tuple right = py::make_tuple(r0 + get_fraction(ur, lr, level), c1);

                switch (square_case) {
                    case 1:
                        // top to left
                        segments.push_back(py::make_tuple(top, left));
                        break;
                    case 2:
                        // right to top
                        segments.push_back(py::make_tuple(right, top));
                        break;
                    case 3:
                        // right to left
                        segments.push_back(py::make_tuple(right, left));
                        break;
                    case 4:
                        // left to bottom
                        segments.push_back(py::make_tuple(left, bottom));
                        break;
                    case 5:
                        // top to bottom
                        segments.push_back(py::make_tuple(top, bottom));
                        break;
                    case 6:
                        segments.push_back(py::make_tuple(right, top));
                        segments.push_back(py::make_tuple(left, bottom));
                        break;
                    case 7:
                        // right to bottom
                        segments.push_back(py::make_tuple(right, bottom));
                        break;
                    case 8:
                        // bottom to right
                        segments.push_back(py::make_tuple(bottom, right));
                        break;
                    case 9:
                        segments.push_back(py::make_tuple(top, left));
                        segments.push_back(py::make_tuple(bottom, right));
                        break;
                    case 10:
                        // bottom to top
                        segments.push_back(py::make_tuple(bottom, top));
                        break;
                    case 11:
                        // bottom to left
                        segments.push_back(py::make_tuple(bottom, left));
                        break;
                    case 12:
                        // lef to right
                        segments.push_back(py::make_tuple(left, right));
                        break;
                    case 13:
                        // top to right
                        segments.push_back(py::make_tuple(top, right));
                        break;
                    case 14:
                        // left to top
                        segments.push_back(py::make_tuple(left, top));
                        break;
                }

                // if (square_case == 1) {
                //     // top to left
                //     segments.push_back(py::make_tuple(top, left));
                // } else if (square_case == 2) {
                //     // right to top
                //     segments.push_back(py::make_tuple(right, top));
                // } else if (square_case == 3) {
                //     // right to left
                //     segments.push_back(py::make_tuple(right, left));
                // } else if (square_case == 4) {
                //     // left to bottom
                //     segments.push_back(py::make_tuple(left, bottom));
                // } else if (square_case == 5) {
                //     // top to bottom
                //     segments.push_back(py::make_tuple(top, bottom));
                // } else if (square_case == 6) {
                //     segments.push_back(py::make_tuple(right, top));
                //     segments.push_back(py::make_tuple(left, bottom));
                // } else if (square_case == 7) {
                //     // right to bottom
                //     segments.push_back(py::make_tuple(right, bottom));
                // } else if (square_case == 8) {
                //     // bottom to right
                //     segments.push_back(py::make_tuple(bottom, right));
                // } else if (square_case == 9) {
                //     segments.push_back(py::make_tuple(top, left));
                //     segments.push_back(py::make_tuple(bottom, right));
                // } else if (square_case == 10) {
                //     // bottom to top
                //     segments.push_back(py::make_tuple(bottom, top));
                // } else if (square_case == 11) {
                //     // bottom to left
                //     segments.push_back(py::make_tuple(bottom, left));
                // } else if (square_case == 12) {
                //     // lef to right
                //     segments.push_back(py::make_tuple(left, right));
                // } else if (square_case == 13) {
                //     // top to right
                //     segments.push_back(py::make_tuple(top, right));
                // } else if (square_case == 14) {
                //     // left to top
                //     segments.push_back(py::make_tuple(left, top));
                // }
            }
        }
        output.push_back(py::cast(segments));
    }
    return output.size();  // py::cast(output);
}
}  // namespace get_contour_segments_stacked
