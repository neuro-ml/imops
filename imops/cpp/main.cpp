#include <cassert>
#include <iostream>
#include <stdexcept>
#include "interp2d/interpolator.h"
#include <Python.h>

PYBIND11_MODULE(cpp_modules, m) {
    py::class_<Interpolator>(m, "Linear2DInterpolatorCpp", "Interpolator class")
        .def(py::init<const Triangulator::pyarr_size_t&, int, std::optional<Triangulator::pyarr_size_t>>())
        .def("__call__", &Interpolator::operator());
}

int main() {}
