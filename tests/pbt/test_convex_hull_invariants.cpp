#include "algoat/geometry/convex_hull.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck.h>

using namespace algoat::geometry;

namespace rc {

template <> struct Arbitrary<Point2D> {
    static Gen<Point2D> arbitrary() {
        return gen::map(
            gen::tuple(gen::inRange<int>(-10000, 10000), gen::inRange<int>(-10000, 10000)),
            [](const std::tuple<int, int>& t) {
                return Point2D{static_cast<double>(std::get<0>(t)) / 10.0,
                               static_cast<double>(std::get<1>(t)) / 10.0};
            });
    }
};

template <> struct Arbitrary<Point3D> {
    static Gen<Point3D> arbitrary() {
        return gen::map(gen::tuple(gen::inRange<int>(-10000, 10000),
                                   gen::inRange<int>(-10000, 10000),
                                   gen::inRange<int>(-10000, 10000)),
                        [](const std::tuple<int, int, int>& t) {
                            return Point3D{static_cast<double>(std::get<0>(t)) / 10.0,
                                           static_cast<double>(std::get<1>(t)) / 10.0,
                                           static_cast<double>(std::get<2>(t)) / 10.0};
                        });
    }
};

} // namespace rc

// ============================================================================
// 1. 2D Convex Hull: Containment & Convexity Invariants
// ============================================================================

TEST(ConvexHullPBT, AllPointsInsideOrOn2DHull) {
    rc::check("All input points lie inside or on the boundary of the computed 2D hull",
              [](const std::vector<Point2D>& pts) {
                  auto hull = convex_hull_2d_andrew(pts);
                  if (hull.vertices.size() < 3) {
                      RC_SUCCEED("Degenerate or collinear points");
                  }

                  // Strict Counter-Clockwise convexity check
                  const std::size_t m = hull.vertices.size();
                  for (std::size_t i = 0; i < m; ++i) {
                      const auto& p0 = hull.vertices[i];
                      const auto& p1 = hull.vertices[(i + 1) % m];
                      const auto& p2 = hull.vertices[(i + 2) % m];
                      RC_ASSERT(orient2d(p0, p1, p2) == Orientation::CounterClockwise);
                  }

                  // Containment check: every point in pts must be on or to the left of each hull
                  // edge
                  for (const auto& pt : pts) {
                      for (std::size_t i = 0; i < m; ++i) {
                          const auto& u = hull.vertices[i];
                          const auto& v = hull.vertices[(i + 1) % m];
                          RC_ASSERT(orient2d(u, v, pt) != Orientation::Clockwise);
                      }
                  }
              });
}

TEST(ConvexHullPBT, AndrewAndChanEquivalence) {
    rc::check(
        "Chan's algorithm produces identical hull vertices and metrics as Andrew's algorithm",
        [](const std::vector<Point2D>& pts) {
            auto hull_andrew = convex_hull_2d_andrew(pts);
            auto hull_chan = convex_hull_2d_chan(pts);

            RC_ASSERT(hull_andrew.vertices.size() == hull_chan.vertices.size());
            RC_ASSERT(std::abs(hull_andrew.area - hull_chan.area) < 1e-9);
            RC_ASSERT(std::abs(hull_andrew.perimeter - hull_chan.perimeter) < 1e-9);

            for (std::size_t i = 0; i < hull_andrew.vertices.size(); ++i) {
                RC_ASSERT(std::abs(hull_andrew.vertices[i].x - hull_chan.vertices[i].x) < 1e-12);
                RC_ASSERT(std::abs(hull_andrew.vertices[i].y - hull_chan.vertices[i].y) < 1e-12);
            }
        });
}

// ============================================================================
// 2. 3D Quickhull: Containment & Non-Negative Volume Invariants
// ============================================================================

TEST(ConvexHullPBT, AllPointsInsideOrOn3DPolyhedron) {
    rc::check("All input points lie inside or on the boundary of the computed 3D polyhedron",
              [](const std::vector<Point3D>& pts) {
                  auto poly = convex_hull_3d_quickhull(pts);

                  RC_ASSERT(poly.volume >= 0.0);
                  RC_ASSERT(poly.surface_area >= 0.0);

                  if (poly.triangle_facets.empty()) {
                      RC_SUCCEED("Degenerate 0D/1D/2D point set");
                  }

                  // Euler characteristic check for triangulated 2-sphere mesh: F = 2V - 4
                  const std::size_t V = poly.vertices.size();
                  const std::size_t F = poly.triangle_facets.size();
                  RC_ASSERT(F == 2 * V - 4);

                  // Containment check: signed distance of every point to every facet <= 1e-6
                  for (const auto& tri : poly.triangle_facets) {
                      const auto& v0 = poly.vertices[tri[0]];
                      const auto& v1 = poly.vertices[tri[1]];
                      const auto& v2 = poly.vertices[tri[2]];

                      const double ax = v1.x - v0.x, ay = v1.y - v0.y, az = v1.z - v0.z;
                      const double bx = v2.x - v0.x, by = v2.y - v0.y, bz = v2.z - v0.z;

                      const double nx = ay * bz - az * by;
                      const double ny = az * bx - ax * bz;
                      const double nz = ax * by - ay * bx;
                      const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
                      if (norm < 1e-14)
                          continue;

                      const double unx = nx / norm;
                      const double uny = ny / norm;
                      const double unz = nz / norm;

                      for (const auto& p : pts) {
                          const double d =
                              unx * (p.x - v0.x) + uny * (p.y - v0.y) + unz * (p.z - v0.z);
                          RC_ASSERT(d <= 1e-5);
                      }
                  }
              });
}
