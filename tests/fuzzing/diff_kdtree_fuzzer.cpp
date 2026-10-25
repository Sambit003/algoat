#include "algoat/geometry/kdtree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::geometry;

using Point3D = std::array<double, 3>;

static double dist_euclidean_sq(const Point3D& a, const Point3D& b) {
    double sum = 0.0;
    for (std::size_t d = 0; d < 3; ++d) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    constexpr std::size_t point_bytes = 3 * sizeof(double);
    if (size < point_bytes * 2) {
        return 0;
    }

    Point3D query;
    std::memcpy(query.data(), data, point_bytes);
    for (std::size_t d = 0; d < 3; ++d) {
        if (!std::isfinite(query[d])) {
            return 0;
        }
    }

    const std::size_t remaining_bytes = size - point_bytes;
    std::size_t num_points = remaining_bytes / point_bytes;
    if (num_points > 200) {
        num_points = 200; // Bound size for fast fuzz iterations
    }

    std::vector<Point3D> points(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        std::memcpy(points[i].data(), data + point_bytes + (i * point_bytes), point_bytes);
        for (std::size_t d = 0; d < 3; ++d) {
            if (!std::isfinite(points[i][d])) {
                return 0;
            }
        }
    }

    KDTree<Point3D, 3> tree(points);
    if (tree.size() != num_points) {
        __builtin_trap();
    }

    // 1. k-NN search comparison
    const std::size_t k = (num_points >= 5) ? 5 : num_points;
    auto knn = tree.knn_search(query, k);
    if (knn.size() != k) {
        __builtin_trap();
    }

    std::vector<std::pair<double, std::size_t>> bf_knn;
    bf_knn.reserve(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        bf_knn.emplace_back(std::sqrt(dist_euclidean_sq(query, points[i])), i);
    }
    std::sort(bf_knn.begin(), bf_knn.end());

    for (std::size_t i = 0; i < k; ++i) {
        if (std::abs(knn[i].distance - bf_knn[i].first) > 1e-6) {
            __builtin_trap();
        }
        if (i > 0 && knn[i].distance < knn[i - 1].distance) {
            __builtin_trap();
        }
    }

    // 2. Radius search comparison
    double radius = (knn.empty()) ? 1.0 : knn.front().distance;
    auto rad = tree.radius_search(query, radius);
    std::vector<std::size_t> bf_rad;
    const double radius_sq = radius * radius;
    for (std::size_t i = 0; i < num_points; ++i) {
        if (dist_euclidean_sq(query, points[i]) <= radius_sq) {
            bf_rad.push_back(i);
        }
    }
    std::sort(bf_rad.begin(), bf_rad.end());
    if (rad != bf_rad) {
        __builtin_trap();
    }

    // 3. Box search comparison
    Point3D min_c{query[0] - 1.0, query[1] - 1.0, query[2] - 1.0};
    Point3D max_c{query[0] + 1.0, query[1] + 1.0, query[2] + 1.0};
    auto box = tree.box_search(min_c, max_c);
    std::vector<std::size_t> bf_box;
    for (std::size_t i = 0; i < num_points; ++i) {
        if (points[i][0] >= min_c[0] && points[i][0] <= max_c[0] && points[i][1] >= min_c[1] &&
            points[i][1] <= max_c[1] && points[i][2] >= min_c[2] && points[i][2] <= max_c[2]) {
            bf_box.push_back(i);
        }
    }
    std::sort(bf_box.begin(), bf_box.end());
    if (box != bf_box) {
        __builtin_trap();
    }

    return 0;
}
