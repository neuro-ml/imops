#include "triangulator.h"

class Interpolator {

public:
    using pyarr_double = py::array_t<double, py::array::c_style | py::array::forcecast>;
    const Triangulator triangulation;
    Interpolator(const Triangulator::pyarr_size_t& pypoints, int n_jobs,
                 std::optional<Triangulator::pyarr_size_t> pytriangles)
        : triangulation(pypoints, n_jobs, pytriangles) {
    }

    pyarr_double operator()(const Triangulator::pyarr_size_t& int_points,
                            const pyarr_double& values, const Triangulator::pyarr_size_t& neighbors,
                            double fill_value = 0.0) {
        if (static_cast<long>(triangulation.points.size() / 2) != values.shape(0)) {
            throw std::invalid_argument("Length mismatch between known points and their values");
        }
        if (neighbors.shape(0) != int_points.shape(0)) {
            throw std::invalid_argument("Length mismatch between int_points and their neighbors");
        }

        size_t n = int_points.shape(0);
        std::vector<double> int_values(n);

        omp_set_dynamic(0);  // Explicitly disable dynamic teams
        omp_set_num_threads(triangulation.n_jobs_);

        #pragma parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t x = int_points.at(i, 0);
            size_t y = int_points.at(i, 1);
            auto location_info = triangulation.locate_point(x, y, neighbors.at(i));
            if (location_info.has_value()) {
                size_t t = location_info->first;
                auto& coords_info = location_info->second;
                double one_over_2area = 1.0 / static_cast<double>(coords_info[3]);
                double lambda_1 = static_cast<double>(coords_info[0]);
                double lambda_2 = static_cast<double>(coords_info[1]);
                double lambda_3 = static_cast<double>(coords_info[2]);
                double f1 = values.at(triangulation.triangles.at(t));
                double f2 = values.at(triangulation.triangles.at(t + 1));
                double f3 = values.at(triangulation.triangles.at(t + 2));
                int_values[i] = (f1 * lambda_1 + f2 * lambda_2 + f3 * lambda_3) * one_over_2area;
            } else {
                int_values[i] = fill_value;
            }
        }

        return {{n}, {sizeof(double)}, int_values.data()};
    }
};
