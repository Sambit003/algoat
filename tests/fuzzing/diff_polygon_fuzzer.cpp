#include "algoat/geometry/polygon.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::geometry;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    constexpr std::size_t min_bytes = 8 * sizeof(double); // At least 3 poly points + 1 query point
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

    // Partition doubles: first 2 * num_poly for polygon, remaining for queries
    const std::size_t max_poly_pts = std::min((num_doubles - 2) / 2, std::size_t{64});
    if (max_poly_pts < 3) {
        return 0;
    }

    std::vector<Point2D> poly_pts;
    poly_pts.reserve(max_poly_pts);
    for (std::size_t i = 0; i < max_poly_pts; ++i) {
        poly_pts.push_back({buf[2 * i], buf[2 * i + 1]});
    }

    Polygon poly(poly_pts);

    std::size_t offset = 2 * max_poly_pts;
    std::size_t num_queries = (num_doubles - offset) / 2;
    if (num_queries > 64) {
        num_queries = 64;
    }

    std::vector<Point2D> queries;
    queries.reserve(num_queries);
    for (std::size_t i = 0; i < num_queries; ++i) {
        queries.push_back({buf[offset + 2 * i], buf[offset + 2 * i + 1]});
    }

    const auto& bbox = poly.bounding_box();

    for (const auto& q : queries) {
        auto res_nonzero = poly.contains(q, InclusionRule::NonZeroWinding);
        auto res_evenodd = poly.contains(q, InclusionRule::EvenOdd);

        assert(res_nonzero == PolygonInclusion::Outside ||
               res_nonzero == PolygonInclusion::Inside ||
               res_nonzero == PolygonInclusion::OnBoundary);

        assert(res_evenodd == PolygonInclusion::Outside ||
               res_evenodd == PolygonInclusion::Inside ||
               res_evenodd == PolygonInclusion::OnBoundary);

        if (bbox.is_valid()) {
            const double float_tol_x =
                1e-4 * std::max({1.0, std::abs(static_cast<double>(bbox.min_x)),
                                 std::abs(static_cast<double>(bbox.max_x))});
            const double float_tol_y =
                1e-4 * std::max({1.0, std::abs(static_cast<double>(bbox.min_y)),
                                 std::abs(static_cast<double>(bbox.max_y))});
            if (q.x < static_cast<double>(bbox.min_x) - float_tol_x ||
                q.x > static_cast<double>(bbox.max_x) + float_tol_x ||
                q.y < static_cast<double>(bbox.min_y) - float_tol_y ||
                q.y > static_cast<double>(bbox.max_y) + float_tol_y) {
                assert(res_nonzero == PolygonInclusion::Outside);
                assert(res_evenodd == PolygonInclusion::Outside);
            }
        }
    }

    if (!queries.empty()) {
        std::vector<PolygonInclusion> batch_res_eo(queries.size());
        poly.batch_contains(queries, batch_res_eo, InclusionRule::EvenOdd);
        for (std::size_t i = 0; i < queries.size(); ++i) {
            assert(batch_res_eo[i] == poly.contains(queries[i], InclusionRule::EvenOdd));
        }

        std::vector<PolygonInclusion> batch_res_nz(queries.size());
        poly.batch_contains(queries, batch_res_nz, InclusionRule::NonZeroWinding);
        for (std::size_t i = 0; i < queries.size(); ++i) {
            assert(batch_res_nz[i] == poly.contains(queries[i], InclusionRule::NonZeroWinding));
        }
    }

    return 0;
}
