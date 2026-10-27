/**
 * @file test_delaunay.cpp
 * @brief Unit tests for 2D Delaunay Triangulation and Voronoi Dual.
 */

#include "algoat/geometry/convex_hull.hpp"
#include "algoat/geometry/delaunay.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::geometry;
using algoat::geometry::detail::polygon_area;

// ============================================================================
// 1. Base & Degenerate Cases
// ============================================================================

TEST(Delaunay2DTest, EmptyAndSinglePoint) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> empty;
    dt.build(empty);

    EXPECT_EQ(dt.vertex_count(), 0u);
    EXPECT_EQ(dt.triangle_count(), 0u);
    EXPECT_TRUE(dt.triangles().empty());
    EXPECT_FALSE(dt.locate_point({0.0, 0.0}).has_value());
    EXPECT_EQ(dt.nearest_site({0.0, 0.0}), INVALID_INDEX);

    std::vector<Point2D> single{{1.0, 2.0}};
    dt.build(single);
    EXPECT_EQ(dt.vertex_count(), 1u);
    EXPECT_EQ(dt.triangle_count(), 0u);
    EXPECT_TRUE(dt.triangles().empty());
    EXPECT_EQ(dt.nearest_site({1.0, 2.0}), 0u);
    EXPECT_EQ(dt.nearest_site({10.0, 10.0}), 0u);
}

TEST(Delaunay2DTest, TwoPoints) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {3.0, 4.0}};
    dt.build(pts);

    EXPECT_EQ(dt.vertex_count(), 2u);
    EXPECT_EQ(dt.triangle_count(), 0u);
    EXPECT_TRUE(dt.triangles().empty());
    EXPECT_EQ(dt.nearest_site({0.1, 0.1}), 0u);
    EXPECT_EQ(dt.nearest_site({2.9, 3.9}), 1u);
}

TEST(Delaunay2DTest, CollinearPoints) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    dt.build(pts);

    EXPECT_EQ(dt.vertex_count(), 4u);
    EXPECT_EQ(dt.triangle_count(), 0u);
    EXPECT_TRUE(dt.triangles().empty());
}

TEST(Delaunay2DTest, DuplicatePoints) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}};
    dt.build(pts);

    EXPECT_GE(dt.triangle_count(), 1u);
    EXPECT_FALSE(dt.triangles().empty());
}

// ============================================================================
// 2. Simple Triangulation & Adjacency
// ============================================================================

TEST(Delaunay2DTest, SingleTriangle) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}};
    dt.build(pts);

    ASSERT_EQ(dt.triangle_count(), 1u);
    const auto& t = dt.triangles()[0];
    EXPECT_EQ(t.ab_opp, INVALID_INDEX);
    EXPECT_EQ(t.bc_opp, INVALID_INDEX);
    EXPECT_EQ(t.ca_opp, INVALID_INDEX);

    // CCW orientation check
    EXPECT_EQ(orient2d(pts[t.a], pts[t.b], pts[t.c]), Orientation::CounterClockwise);

    // Circumcenter of right isosceles triangle at (1, 1)
    EXPECT_NEAR(t.circumcenter.x, 1.0, 1e-12);
    EXPECT_NEAR(t.circumcenter.y, 1.0, 1e-12);
    EXPECT_NEAR(t.circumradius, std::sqrt(2.0), 1e-12);
}

TEST(Delaunay2DTest, UnitSquareTwoTriangles) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
    dt.build(pts);

    ASSERT_EQ(dt.triangle_count(), 2u);
    const auto trs = dt.triangles();

    for (const auto& t : trs) {
        EXPECT_EQ(orient2d(pts[t.a], pts[t.b], pts[t.c]), Orientation::CounterClockwise);
    }

    // Exactly one internal shared edge between the two triangles
    uint32_t shared_count_0 = (trs[0].ab_opp != INVALID_INDEX) + (trs[0].bc_opp != INVALID_INDEX) +
                              (trs[0].ca_opp != INVALID_INDEX);
    uint32_t shared_count_1 = (trs[1].ab_opp != INVALID_INDEX) + (trs[1].bc_opp != INVALID_INDEX) +
                              (trs[1].ca_opp != INVALID_INDEX);
    EXPECT_EQ(shared_count_0, 1u);
    EXPECT_EQ(shared_count_1, 1u);
}

// ============================================================================
// 3. Point Location & Nearest Site
// ============================================================================

TEST(Delaunay2DTest, LocatePoint) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}};
    dt.build(pts);

    // Interior query
    auto loc1 = dt.locate_point({0.5, 0.5});
    ASSERT_TRUE(loc1.has_value());
    EXPECT_LT(*loc1, dt.triangle_count());

    // Exterior query
    auto loc_ext = dt.locate_point({5.0, 5.0});
    EXPECT_FALSE(loc_ext.has_value());

    auto loc_ext_neg = dt.locate_point({-1.0, -1.0});
    EXPECT_FALSE(loc_ext_neg.has_value());
}

TEST(Delaunay2DTest, NearestSiteQuery) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}, {5.0, 5.0}};
    dt.build(pts);

    EXPECT_EQ(dt.nearest_site({0.1, 0.1}), 0u);
    EXPECT_EQ(dt.nearest_site({9.9, 0.2}), 1u);
    EXPECT_EQ(dt.nearest_site({9.8, 9.9}), 2u);
    EXPECT_EQ(dt.nearest_site({0.2, 9.7}), 3u);
    EXPECT_EQ(dt.nearest_site({5.1, 4.9}), 4u);
}

// ============================================================================
// 4. Delaunay Circumcircle Empty-Interior Invariant
// ============================================================================

TEST(Delaunay2DTest, DelaunayCircumcircleInvariantUniformRandom) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    const std::size_t N = 200;
    std::vector<Point2D> pts;
    pts.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        pts.push_back({dist(rng), dist(rng)});
    }

    DelaunayTriangulation2D dt;
    dt.build(pts);

    ASSERT_GT(dt.triangle_count(), 0u);
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
            EXPECT_NE(loc, CircleLocation::Inside)
                << "Point " << i << " violates Delaunay circumcircle property for triangle (" << t.a
                << ", " << t.b << ", " << t.c << ")";
        }
    }
}

// ============================================================================
// 5. Voronoi Dual & Area Conservation
// ============================================================================

TEST(Delaunay2DTest, VoronoiAreaConservationSquare) {
    DelaunayTriangulation2D dt;
    std::vector<Point2D> pts{{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0},
                             {5.0, 5.0}, {2.0, 3.0},  {7.0, 8.0}};
    dt.build(pts);

    auto cells = dt.compute_voronoi_cells();
    ASSERT_EQ(cells.size(), pts.size());

    // Center site {5.0, 5.0} should be interior and bounded
    EXPECT_TRUE(cells[4].is_bounded);

    // Sum of clipped cell areas should equal bounding box area
    // Bounding box spans [0, 10] x [0, 10] -> Area = 100.0
    double total_area = 0.0;
    for (const auto& cell : cells) {
        total_area += polygon_area(cell.polygon_vertices);
    }

    EXPECT_NEAR(total_area, 100.0, 1e-9);
}

TEST(Delaunay2DTest, VoronoiAreaConservationRandomCloud) {
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist(0.0, 50.0);

    const std::size_t N = 100;
    std::vector<Point2D> pts;
    pts.reserve(N);
    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (std::size_t i = 0; i < N; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        pts.push_back({x, y});
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    DelaunayTriangulation2D dt;
    dt.build(pts);

    auto cells = dt.compute_voronoi_cells();
    ASSERT_EQ(cells.size(), pts.size());

    double expected_bbox_area = (max_x - min_x) * (max_y - min_y);
    double total_area = 0.0;
    for (const auto& cell : cells) {
        total_area += polygon_area(cell.polygon_vertices);
    }

    EXPECT_NEAR(total_area, expected_bbox_area, 1e-8);
}
