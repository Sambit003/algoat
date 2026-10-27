/**
 * @file test_delaunay_invariants.cpp
 * @brief RapidCheck Property-Based Tests for Delaunay Triangulation and Voronoi Dual.
 */

#include "algoat/geometry/convex_hull.hpp"
#include "algoat/geometry/delaunay.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::geometry;
using algoat::geometry::detail::polygon_area;

namespace rc {

template <> struct Arbitrary<Point2D> {
    static Gen<Point2D> arbitrary() {
        return gen::map(gen::tuple(gen::inRange<int>(-5000, 5000), gen::inRange<int>(-5000, 5000)),
                        [](const std::tuple<int, int>& t) {
                            return Point2D{static_cast<double>(std::get<0>(t)) / 10.0,
                                           static_cast<double>(std::get<1>(t)) / 10.0};
                        });
    }
};

} // namespace rc

// ============================================================================
// 1. Delaunay Circumcircle Empty-Interior Invariant
// ============================================================================

TEST(DelaunayPBT, EmptyCircumcircleInvariant) {
    rc::check(
        "Delaunay empty circumcircle invariant holds for all triangles and non-incident points",
        [](const std::vector<Point2D>& pts) {
            DelaunayTriangulation2D dt;
            dt.build(pts);

            if (dt.triangle_count() == 0) {
                RC_SUCCEED("Collinear or degenerate points");
            }

            const auto trs = dt.triangles();
            for (const auto& t : trs) {
                const auto& pa = pts[t.a];
                const auto& pb = pts[t.b];
                const auto& pc = pts[t.c];

                for (std::size_t i = 0; i < pts.size(); ++i) {
                    if (i == t.a || i == t.b || i == t.c) {
                        continue;
                    }
                    CircleLocation loc = incircle(pa, pb, pc, pts[i]);
                    RC_ASSERT(loc != CircleLocation::Inside);
                }
            }
        });
}

// ============================================================================
// 2. Nearest Site Invariant
// ============================================================================

TEST(DelaunayPBT, NearestSiteMatchesBruteForce) {
    rc::check("nearest_site query returns site with minimal Euclidean distance",
              [](const std::vector<Point2D>& pts, const Point2D& query) {
                  if (pts.empty()) {
                      RC_SUCCEED("Empty points");
                  }

                  DelaunayTriangulation2D dt;
                  dt.build(pts);

                  const uint32_t nearest = dt.nearest_site(query);
                  RC_ASSERT(nearest < pts.size());

                  const double reported_dist_sq =
                      (pts[nearest].x - query.x) * (pts[nearest].x - query.x) +
                      (pts[nearest].y - query.y) * (pts[nearest].y - query.y);

                  for (const auto& pt : pts) {
                      const double d_sq =
                          (pt.x - query.x) * (pt.x - query.x) + (pt.y - query.y) * (pt.y - query.y);
                      RC_ASSERT(reported_dist_sq <= d_sq + 1e-9);
                  }
              });
}

// ============================================================================
// 3. Voronoi Area Conservation Invariant
// ============================================================================

TEST(DelaunayPBT, VoronoiAreaConservation) {
    rc::check("Sum of Voronoi cell areas equals bounding box area",
              [](const std::vector<Point2D>& pts) {
                  if (pts.size() < 3) {
                      RC_SUCCEED("Too few points");
                  }

                  DelaunayTriangulation2D dt;
                  dt.build(pts);

                  if (dt.triangle_count() == 0) {
                      RC_SUCCEED("Collinear points");
                  }

                  double min_x = pts[0].x, max_x = pts[0].x;
                  double min_y = pts[0].y, max_y = pts[0].y;
                  for (const auto& p : pts) {
                      min_x = std::min(min_x, p.x);
                      max_x = std::max(max_x, p.x);
                      min_y = std::min(min_y, p.y);
                      max_y = std::max(max_y, p.y);
                  }

                  const double expected_bbox_area = (max_x - min_x) * (max_y - min_y);
                  if (expected_bbox_area < 1e-8) {
                      RC_SUCCEED("Flat bounding box");
                  }

                  const auto cells = dt.compute_voronoi_cells();
                  RC_ASSERT(cells.size() == pts.size());

                  double total_area = 0.0;
                  for (const auto& cell : cells) {
                      total_area += polygon_area(cell.polygon_vertices);
                  }

                  RC_ASSERT(std::abs(total_area - expected_bbox_area) < 1e-7);
              });
}
