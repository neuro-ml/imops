#include <vector>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../delaunator/delaunator-header-only.hpp"

namespace convex_hull {
namespace py = pybind11;

void parallel_convex_hull_along_axes(
    py::array_t<bool, py::array::c_style | py::array::forcecast> mask,
    const py::tuple& pixel_spacing, double max_dist_from_lungs_mm = 3.0) {
}
}  // namespace convex_hull
