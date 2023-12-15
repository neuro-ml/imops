#include "triangulator.h"

class Interpolator {

public:
    using pyarr_float = py::array_t<float, py::array::c_style | py::array::forcecast>;
    const Triangulator triangulation;
    Interpolator(const Triangulator::pyarr_size_t& points, int n_jobs)
        : triangulation(points, n_jobs) {
    }

    pyarr_float operator()(const Triangulator::pyarr_size_t& int_points, const pyarr_float& values,
                           const Triangulator::pyarr_size_t& neighbors, double fill_value = 0.0) {
        if (triangulation.points.size() / 2 != values.shape()[0]) {
            throw std::invalid_argument("Length mismatch between known points and their values");
        }
        if (neighbors.shape()[0] != int_points.shape()[0]) {
            throw std::invalid_argument("Length mismatch between int_points and their neighbors");
        }

        size_t n = int_points.shape()[0];
        std::vector<float> int_values(n, fill_value);

        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(triangulation.n_jobs_);

        #pragma parallel for
        for (size_t i = 0; i < n; ++i) {
            size_t x = int_points.at(i, 0);
            size_t y = int_points.at(i, 1);
            auto location_info = triangulation.locate_point(x, y, neighbors.at(i));
            if (location_info != std::nullopt) {
                size_t t = location_info->first;
                auto& coords_info = location_info->second;
                double one_over_2area = 1.0 / static_cast<double>(coords_info[3]);
                double lambda_1 = static_cast<double>(coords_info[0]) * one_over_2area;
                double lambda_2 = static_cast<double>(coords_info[1]) * one_over_2area;
                double lambda_3 = static_cast<double>(coords_info[2]) * one_over_2area;
                double f1 = values.at(triangulation.triangles.at(t));  // implicit float -> double
                double f2 = values.at(triangulation.triangles.at(t + 1));
                double f3 = values.at(triangulation.triangles.at(t + 2));
                int_values[i] = static_cast<float>(f1 * lambda_1 + f2 * lambda_2 + f3 * lambda_3);
            }
        }
        
        return {{n}, {sizeof(float)}, int_values.data()};
    }
};
