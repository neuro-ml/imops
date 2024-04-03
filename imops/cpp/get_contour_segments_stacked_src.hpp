#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <iostream>

namespace get_contour_segments_stacked {

struct Point {
    float x;
    float y;
};

struct Segment {
    Point a;
    Point b;
};

namespace py = pybind11;
using pyarr_float = py::array_t<float>;

inline float get_fraction(float from_value, float to_value, float level) {
    return to_value == from_value ? 0 : ((level - from_value) / (to_value - from_value));
}

py::list get_contour_segments_stacked_func(const py::array_t<float>& array_input, float level) {
    std::vector<py::list> output;
    auto array = array_input.unchecked<3>();
    int dim_Z = array.shape(0);
    int dim_R = array.shape(1);
    int dim_C = array.shape(2);

    for (int z_index = 0; z_index < dim_Z; ++z_index) {
        std::vector<py::tuple> segments;

        for (int r0 = 0; r0 < dim_R - 1; ++r0) {
            for (int c0 = 0; c0 < dim_C - 1; ++c0) {

                int r1 = r0 + 1;
                int c1 = c0 + 1;

                float ul = array(z_index, r0, c0);
                float ur = array(z_index, r0, c1);
                float ll = array(z_index, r1, c0);
                float lr = array(z_index, r1, c1);

                if (std::isnan(ul) or std::isnan(ur) or std::isnan(ll) or std::isnan(lr)) {
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

                if (square_case == 1) {
                    // top to left
                    segments.push_back(py::make_tuple(top, left));
                } else if (square_case == 2) {
                    // right to top
                    segments.push_back(py::make_tuple(right, top));
                } else if (square_case == 3) {
                    // right to left
                    segments.push_back(py::make_tuple(right, left));
                } else if (square_case == 4) {
                    // left to bottom
                    segments.push_back(py::make_tuple(left, bottom));
                } else if (square_case == 5) {
                    // top to bottom
                    segments.push_back(py::make_tuple(top, bottom));
                } else if (square_case == 6) {
                    segments.push_back(py::make_tuple(right, top));
                    segments.push_back(py::make_tuple(left, bottom));
                } else if (square_case == 7) {
                    // right to bottom
                    segments.push_back(py::make_tuple(right, bottom));
                } else if (square_case == 8) {
                    // bottom to right
                    segments.push_back(py::make_tuple(bottom, right));
                } else if (square_case == 9) {
                    segments.push_back(py::make_tuple(top, left));
                    segments.push_back(py::make_tuple(bottom, right));
                } else if (square_case == 10) {
                    // bottom to top
                    segments.push_back(py::make_tuple(bottom, top));
                } else if (square_case == 11) {
                    // bottom to left
                    segments.push_back(py::make_tuple(bottom, left));
                } else if (square_case == 12) {
                    // lef to right
                    segments.push_back(py::make_tuple(left, right));
                } else if (square_case == 13) {
                    // top to right
                    segments.push_back(py::make_tuple(top, right));
                } else if (square_case == 14) {
                    // left to top
                    segments.push_back(py::make_tuple(left, top));
                }
            }
        }
        output.push_back(py::cast(std::move(segments)));
    }
    return py::cast(std::move(output));
}
}  // namespace get_contour_segments_stacked
