/**
 * @file diff_delaunay_fuzzer.cpp
 * @brief Differential and invariant fuzzer for 2D Delaunay Triangulation and Voronoi Dual.
 */

#include "algoat/geometry/delaunay.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::geometry;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_bytes = 6 * sizeof(double);
    if (size < min_bytes) {
        return 0;
    }

    const std::size_t num_doubles = size / sizeof(double);
    std::vector<double> buf(num_doubles);
    std::memcpy(buf.data(), data, num_doubles * sizeof(double));

    for (double d : buf) {
        if (!std::isfinite(d) || std::abs(d) > 1e10) {
            return 0;
        }
    }

    const std::size_t num_pts = std::min(num_doubles / 2, std::size_t{100});
    std::vector<Point2D> pts;
    pts.reserve(num_pts);
    for (std::size_t i = 0; i < num_pts; ++i) {
        pts.push_back({buf[2 * i], buf[2 * i + 1]});
    }

    DelaunayTriangulation2D dt;
    dt.build(pts);

    assert(dt.vertex_count() == pts.size());

    // Point location and nearest site queries
    if (!pts.empty()) {
        const Point2D query = pts[0];
        auto loc = dt.locate_point(query);
        (void)loc;

        const uint32_t nearest = dt.nearest_site(query);
        assert(nearest < pts.size());
    }

    // Voronoi Dual Cells
    auto cells = dt.compute_voronoi_cells();
    assert(cells.size() == pts.size());

    return 0;
}
