#include "algoat/geometry/convex_hull.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::geometry;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_bytes = 10 * sizeof(double);
    if (size < min_bytes) {
        return 0;
    }

    const std::size_t num_doubles = size / sizeof(double);
    std::vector<double> buf(num_doubles);
    std::memcpy(buf.data(), data, num_doubles * sizeof(double));

    for (double d : buf) {
        if (!std::isfinite(d) || std::abs(d) > 1e12) {
            return 0;
        }
    }

    // 1. Fuzz 2D Convex Hull
    const std::size_t num_pts_2d = std::min(num_doubles / 2, std::size_t{100});
    std::vector<Point2D> pts_2d;
    pts_2d.reserve(num_pts_2d);
    for (std::size_t i = 0; i < num_pts_2d; ++i) {
        pts_2d.push_back({buf[2 * i], buf[2 * i + 1]});
    }

    auto hull_andrew = convex_hull_2d_andrew(pts_2d);
    auto hull_chan = convex_hull_2d_chan(pts_2d);

    assert(hull_andrew.vertices.size() == hull_chan.vertices.size());
    assert(hull_andrew.area >= 0.0);
    assert(hull_andrew.perimeter >= 0.0);

    // 2. Fuzz 3D Quickhull
    const std::size_t num_pts_3d = std::min(num_doubles / 3, std::size_t{100});
    std::vector<Point3D> pts_3d;
    pts_3d.reserve(num_pts_3d);
    for (size_t i = 0; i < num_pts_3d; ++i) {
        pts_3d.push_back({buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]});
    }

    auto poly = convex_hull_3d_quickhull(pts_3d);
    assert(poly.volume >= 0.0);
    assert(poly.surface_area >= 0.0);

    return 0;
}
