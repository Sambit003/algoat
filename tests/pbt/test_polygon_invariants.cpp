#include "algoat/geometry/polygon.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::geometry;

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
// 1. Polygon Bounding Box Enclosure Invariant
// ============================================================================

TEST(PolygonPBT, VerticesEnclosedByBoundingBox) {
    rc::check("All polygon vertices lie within its bounding box",
              [](const std::vector<Point2D>& raw_pts) {
                  if (raw_pts.size() < 3) {
                      RC_SUCCEED("Degenerate");
                  }

                  Polygon poly(raw_pts);
                  const auto& bbox = poly.bounding_box();
                  RC_ASSERT(bbox.is_valid());

                  constexpr float eps = 1e-4f;
                  for (std::size_t i = 0; i < poly.edge_count(); ++i) {
                      RC_ASSERT(raw_pts[i].x >= bbox.min_x - eps);
                      RC_ASSERT(raw_pts[i].x <= bbox.max_x + eps);
                      RC_ASSERT(raw_pts[i].y >= bbox.min_y - eps);
                      RC_ASSERT(raw_pts[i].y <= bbox.max_y + eps);
                  }
              });
}

// ============================================================================
// 2. Points Outside Bounding Box are Strictly Outside
// ============================================================================

TEST(PolygonPBT, PointsOutsideBBoxAreOutside) {
    rc::check("Points outside the polygon bounding box always return Outside",
              [](const std::vector<Point2D>& raw_pts, const Point2D& query) {
                  if (raw_pts.size() < 3) {
                      RC_SUCCEED("Degenerate");
                  }

                  Polygon poly(raw_pts);
                  const auto& bbox = poly.bounding_box();
                  if (!bbox.is_valid()) {
                      RC_SUCCEED("Invalid bbox");
                  }

                  const bool strictly_outside = query.x < bbox.min_x || query.x > bbox.max_x ||
                                                query.y < bbox.min_y || query.y > bbox.max_y;

                  if (strictly_outside) {
                      RC_ASSERT(poly.contains(query, InclusionRule::NonZeroWinding) ==
                                PolygonInclusion::Outside);
                      RC_ASSERT(poly.contains(query, InclusionRule::EvenOdd) ==
                                PolygonInclusion::Outside);
                  }
              });
}

// ============================================================================
// 3. Vertices & Edge Midpoints Always OnBoundary
// ============================================================================

TEST(PolygonPBT, VerticesAndEdgeMidpointsAreOnBoundary) {
    rc::check("Vertices and edge midpoints return OnBoundary",
              [](const std::vector<Point2D>& raw_pts) {
                  if (raw_pts.size() < 3) {
                      RC_SUCCEED("Degenerate");
                  }

                  Polygon poly(raw_pts);
                  if (poly.edge_count() < 3) {
                      RC_SUCCEED("Degenerate after deduplication");
                  }

                  for (std::size_t i = 0; i < poly.edge_count(); ++i) {
                      const auto& p1 = raw_pts[i];
                      const auto& p2 = raw_pts[(i + 1 == poly.edge_count()) ? 0 : i + 1];

                      // Vertex test
                      RC_ASSERT(poly.contains(p1, InclusionRule::NonZeroWinding) ==
                                PolygonInclusion::OnBoundary);
                      RC_ASSERT(poly.contains(p1, InclusionRule::EvenOdd) ==
                                PolygonInclusion::OnBoundary);

                      // Edge midpoint test
                      Point2D mid{0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y)};
                      RC_ASSERT(poly.contains(mid, InclusionRule::NonZeroWinding) ==
                                PolygonInclusion::OnBoundary);
                      RC_ASSERT(poly.contains(mid, InclusionRule::EvenOdd) ==
                                PolygonInclusion::OnBoundary);
                  }
              });
}

// ============================================================================
// 4. Batch vs Single Consistency Invariant
// ============================================================================

TEST(PolygonPBT, BatchMatchesSingleQueryEquivalence) {
    rc::check("Batch containment queries exactly match single containment queries",
              [](const std::vector<Point2D>& raw_pts, const std::vector<Point2D>& queries) {
                  if (raw_pts.size() < 3 || queries.empty()) {
                      RC_SUCCEED("Degenerate");
                  }

                  Polygon poly(raw_pts);
                  std::vector<PolygonInclusion> batch_res(queries.size());

                  for (auto rule : {InclusionRule::NonZeroWinding, InclusionRule::EvenOdd}) {
                      poly.batch_contains(queries, batch_res, rule);

                      for (std::size_t i = 0; i < queries.size(); ++i) {
                          PolygonInclusion single_res = poly.contains(queries[i], rule);
                          RC_ASSERT(batch_res[i] == single_res);
                      }
                  }
              });
}

// ============================================================================
// 5. Regular Convex Polygons Center Point Invariant
// ============================================================================

TEST(PolygonPBT, RegularPolygonCenterIsInside) {
    rc::check("Center of regular convex polygon centered at origin is strictly inside",
              [](int n_sides, int radius_int) {
                  int n = std::clamp(std::abs(n_sides) % 64, 3, 60);
                  double r = static_cast<double>(std::clamp(std::abs(radius_int) % 1000, 1, 1000));

                  std::vector<Point2D> pts;
                  pts.reserve(n);
                  for (int i = 0; i < n; ++i) {
                      double theta =
                          2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n);
                      pts.push_back({r * std::cos(theta), r * std::sin(theta)});
                  }

                  Polygon poly(pts);
                  Point2D center{0.0, 0.0};
                  RC_ASSERT(poly.contains(center, InclusionRule::NonZeroWinding) ==
                            PolygonInclusion::Inside);
                  RC_ASSERT(poly.contains(center, InclusionRule::EvenOdd) ==
                            PolygonInclusion::Inside);
              });
}
