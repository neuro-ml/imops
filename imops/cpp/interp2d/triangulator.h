#include <omp.h>
#include <memory>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "delaunator/delaunator-header-only.hpp"
#include "utils.h"

namespace py = pybind11;

class Triangulator {
private:
    size_t n_jobs_;
    friend class Interpolator;

public:
    using pyarr_size_t = py::array_t<size_t, py::array::c_style | py::array::forcecast>;
    std::vector<size_t> points;
    std::vector<size_t> triangles;
    std::vector<std::vector<size_t>> point2tri;
    std::unordered_map<uint64_t, std::array<size_t, 2>> edge2tri;

    Triangulator(const pyarr_size_t& pypoints, int n_jobs,
                 std::optional<pyarr_size_t> pytriangles) {
        if (n_jobs <= 0 and n_jobs != -1) {
            throw std::invalid_argument(
                "Invalid number of workers, has to be -1 or positive integer");
        }
        if (pytriangles.has_value()) {
            if (pytriangles->shape(1) != 3 or pytriangles->shape(0) == 0 or
                pytriangles->size() / 3 != pytriangles->shape(0)) {
                throw std::invalid_argument("Passed triangles argument has an incorrect shape");
            }
        }

        n_jobs_ = n_jobs == -1 ? omp_get_num_procs() : n_jobs;
        size_t n = pypoints.shape(0);

        points.resize(2 * n);

        omp_set_dynamic(0);  // Explicitly disable dynamic teams
        omp_set_num_threads(n_jobs_);

        if (pytriangles.has_value()) {
            #pragma parallel for
            for (size_t i = 0; i < n; ++i) {
                size_t j = 2 * i;
                points.at(j) = pypoints.at(i, 0);
                points.at(j + 1) = pypoints.at(i, 1);
            }

            size_t m = pytriangles->shape(0);
            triangles.resize(3 * m);

            #pragma parallel for
            for (size_t i = 0; i < m; ++i) {
                size_t j = 3 * i;
                triangles.at(j) = pytriangles->at(i, 0);
                triangles.at(j + 1) = pytriangles->at(i, 1);
                triangles.at(j + 2) = pytriangles->at(i, 2);
            }
        }

        else {
            std::vector<double> double_points(2 * n);
            #pragma parallel for
            for (size_t i = 0; i < n; ++i) {
                double_points.at(2 * i) = static_cast<double>(pypoints.at(i, 0));
                double_points.at(2 * i + 1) = static_cast<double>(pypoints.at(i, 1));
            }
            delaunator::Delaunator delaunated(double_points);
            triangles = std::move(delaunated.triangles);

            #pragma parallel for
            for (size_t i = 0; i < n; ++i) {
                size_t j = 2 * i;
                points.at(j) = static_cast<size_t>(delaunated.coords.at(j));
                points.at(j + 1) = static_cast<size_t>(delaunated.coords.at(j + 1));
            }
        }

        point2tri.resize(n);
        for (size_t i = 0; i < triangles.size() / 3; ++i) {
            size_t t = 3 * i;
            for (size_t j = 0; j < 3; ++j) {
                size_t a = triangles.at(t + j);
                size_t b = triangles.at(t + fast_mod(j + 1, 3));
                point2tri.at(a).push_back(t);
                uint64_t e = a < b ? elegant_pair(a, b) : elegant_pair(b, a);
                auto got = edge2tri.find(e);
                if (got != edge2tri.end()) {
                    got->second.at(1) = t;
                } else {
                    edge2tri[e] = {t, t};
                }
            }
        }
    }

    inline std::optional<std::pair<size_t, std::array<int64_t, 4>>> locate_point(
        size_t x, size_t y, size_t neighbor) const {
        size_t nx = points.at(2 * neighbor);
        size_t ny = points.at(2 * neighbor + 1);
        int64_t curr_t = -1;
        size_t curr_a, curr_b;
        for (size_t t : point2tri.at(neighbor)) {
            auto coords_info = barycentric_coords(x, y, t);
            if (point_in_triangle(coords_info)) {
                return {std::make_pair(t, coords_info)};
            }
            for (size_t j = 0; j < 3; ++j) {
                size_t a = triangles.at(t + j);
                size_t b = triangles.at(t + fast_mod(j + 1, 3));
                if (a == neighbor or b == neighbor) {
                    continue;
                }
                size_t ax = points.at(2 * a);
                size_t ay = points.at(2 * a + 1);
                size_t bx = points.at(2 * b);
                size_t by = points.at(2 * b + 1);
                if (segments_intersection(nx, ny, x, y, ax, ay, bx, by)) {
                    curr_t = static_cast<int64_t>(t);
                    curr_a = a;
                    curr_b = b;
                    if (curr_a > curr_b) {
                        std::swap(curr_a, curr_b);
                    }
                    break;
                }
            }
            if (curr_t != -1) {
                break;
            }
        }

        if (curr_t == -1) {
            return std::nullopt;
        }

        while (true) {
            uint64_t e = elegant_pair(curr_a, curr_b);  // already curr_a < curr_b
            auto& adj_t = edge2tri.at(e);
            if (adj_t.at(0) == adj_t.at(1)) {
                return std::nullopt;
            }
            size_t t = (adj_t.at(0) == static_cast<size_t>(curr_t)) ? adj_t.at(1) : adj_t.at(0);
            auto coords_info = barycentric_coords(x, y, t);
            if (point_in_triangle(coords_info)) {
                return {std::make_pair(t, coords_info)};
            }
            for (size_t i = 0; i < 3; ++i) {
                size_t a = triangles.at(t + i);
                size_t b = triangles.at(t + fast_mod(i + 1, 3));
                if (a > b) {
                    std::swap(a, b);
                }
                if (a == curr_a and b == curr_b) {
                    continue;
                }
                size_t ax = points.at(2 * a);
                size_t ay = points.at(2 * a + 1);
                size_t bx = points.at(2 * b);
                size_t by = points.at(2 * b + 1);
                if (segments_intersection(nx, ny, x, y, ax, ay, bx, by)) {
                    curr_t = static_cast<int64_t>(t);
                    curr_a = a;
                    curr_b = b;
                    break;
                }
            }
        }
    }

    inline std::array<int64_t, 4> barycentric_coords(size_t x, size_t y, size_t t) const {
        size_t r1 = 2 * triangles.at(t);
        auto x1 = static_cast<int64_t>(points.at(r1));
        auto y1 = static_cast<int64_t>(points.at(r1 + 1));
        size_t r2 = 2 * triangles.at(t + 1);
        auto x2 = static_cast<int64_t>(points.at(r2));
        auto y2 = static_cast<int64_t>(points.at(r2 + 1));
        size_t r3 = 2 * triangles.at(t + 2);
        auto x3 = static_cast<int64_t>(points.at(r3));
        auto y3 = static_cast<int64_t>(points.at(r3 + 1));
        auto x1y2 = x1 * y2, x1y3 = x1 * y3;
        auto x2y3 = x2 * y3, x2y1 = x2 * y1;
        auto x3y1 = x3 * y1, x3y2 = x3 * y2;
        int64_t xx = static_cast<int64_t>(x);
        int64_t yy = static_cast<int64_t>(y);
        std::array<int64_t, 4> coords_info;
        coords_info[0] = x2y3 - x3y2 + (y2 - y3) * xx + (x3 - x2) * yy;
        coords_info[1] = x3y1 - x1y3 + (y3 - y1) * xx + (x1 - x3) * yy;
        coords_info[2] = x1y2 - x2y1 + (y1 - y2) * xx + (x2 - x1) * yy;
        coords_info[3] = x1y2 - x1y3 + x2y3 - x2y1 + x3y1 - x3y2;
        return coords_info;
    }
};
