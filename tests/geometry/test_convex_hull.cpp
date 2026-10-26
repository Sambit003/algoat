#include "algoat/geometry/convex_hull.hpp"
#include "algoat/geometry/predicates.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::geometry;

// ============================================================================
// 1. Andrew's Monotone Chain 2D: Base & Degenerate Cases
// ============================================================================

TEST(ConvexHull2DTest, EmptyAndSinglePoint) {
    std::vector<Point2D> empty;
    auto hull_empty = convex_hull_2d_andrew(empty);
    EXPECT_TRUE(hull_empty.vertices.empty());
    EXPECT_DOUBLE_EQ(hull_empty.area, 0.0);
    EXPECT_DOUBLE_EQ(hull_empty.perimeter, 0.0);

    std::vector<Point2D> single{{1.0, 2.0}};
    auto hull_single = convex_hull_2d_andrew(single);
    ASSERT_EQ(hull_single.vertices.size(), 1u);
    EXPECT_DOUBLE_EQ(hull_single.vertices[0].x, 1.0);
    EXPECT_DOUBLE_EQ(hull_single.vertices[0].y, 2.0);
    EXPECT_DOUBLE_EQ(hull_single.area, 0.0);
    EXPECT_DOUBLE_EQ(hull_single.perimeter, 0.0);
}

TEST(ConvexHull2DTest, TwoPoints) {
    std::vector<Point2D> pts{{0.0, 0.0}, {3.0, 4.0}};
    auto hull = convex_hull_2d_andrew(pts);
    ASSERT_EQ(hull.vertices.size(), 2u);
    EXPECT_DOUBLE_EQ(hull.area, 0.0);
    EXPECT_DOUBLE_EQ(hull.perimeter, 10.0); // 2 * distance(p0, p1) = 2 * 5 = 10
}

TEST(ConvexHull2DTest, DuplicatePoints) {
    std::vector<Point2D> pts{{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}};
    auto hull = convex_hull_2d_andrew(pts);
    ASSERT_EQ(hull.vertices.size(), 1u);
    EXPECT_DOUBLE_EQ(hull.vertices[0].x, 1.0);
    EXPECT_DOUBLE_EQ(hull.vertices[0].y, 1.0);
    EXPECT_DOUBLE_EQ(hull.area, 0.0);
}

TEST(ConvexHull2DTest, CollinearPoints) {
    std::vector<Point2D> pts{{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    auto hull = convex_hull_2d_andrew(pts);
    // Strict convex hull for collinear points returns extreme endpoints
    ASSERT_EQ(hull.vertices.size(), 2u);
    EXPECT_DOUBLE_EQ(hull.vertices[0].x, 0.0);
    EXPECT_DOUBLE_EQ(hull.vertices[0].y, 0.0);
    EXPECT_DOUBLE_EQ(hull.vertices[1].x, 3.0);
    EXPECT_DOUBLE_EQ(hull.vertices[1].y, 3.0);
    EXPECT_DOUBLE_EQ(hull.area, 0.0);
}

// ============================================================================
// 2. Andrew's Monotone Chain 2D: Standard Polygons
// ============================================================================

TEST(ConvexHull2DTest, UnitSquareWithInteriorPoints) {
    std::vector<Point2D> pts{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0},
                             {0.5, 0.5}, {0.2, 0.8}, {0.7, 0.3}};
    auto hull = convex_hull_2d_andrew(pts);
    ASSERT_EQ(hull.vertices.size(), 4u);
    EXPECT_NEAR(hull.area, 1.0, 1e-12);
    EXPECT_NEAR(hull.perimeter, 4.0, 1e-12);

    // Verify counter-clockwise orientation
    for (std::size_t i = 0; i < hull.vertices.size(); ++i) {
        const auto& p0 = hull.vertices[i];
        const auto& p1 = hull.vertices[(i + 1) % hull.vertices.size()];
        const auto& p2 = hull.vertices[(i + 2) % hull.vertices.size()];
        EXPECT_EQ(orient2d(p0, p1, p2), Orientation::CounterClockwise);
    }
}

TEST(ConvexHull2DTest, CollinearPointsOnBoundary) {
    // Square with intermediate points on edges
    std::vector<Point2D> pts{{0.0, 0.0}, {0.5, 0.0}, {1.0, 0.0}, {1.0, 0.5}, {1.0, 1.0},
                             {0.5, 1.0}, {0.0, 1.0}, {0.0, 0.5}, {0.5, 0.5}};
    auto hull = convex_hull_2d_andrew(pts);
    // Strict convex hull must pop intermediate collinear boundary points
    ASSERT_EQ(hull.vertices.size(), 4u);
    EXPECT_NEAR(hull.area, 1.0, 1e-12);
    EXPECT_NEAR(hull.perimeter, 4.0, 1e-12);
}

// ============================================================================
// 3. Chan's Output-Sensitive 2D Convex Hull
// ============================================================================

TEST(ConvexHull2DTest, ChanBaseAndDegenerateCases) {
    std::vector<Point2D> empty;
    auto hull_empty = convex_hull_2d_chan(empty);
    EXPECT_TRUE(hull_empty.vertices.empty());

    std::vector<Point2D> single{{5.0, 5.0}};
    auto hull_single = convex_hull_2d_chan(single);
    ASSERT_EQ(hull_single.vertices.size(), 1u);
    EXPECT_DOUBLE_EQ(hull_single.vertices[0].x, 5.0);
    EXPECT_DOUBLE_EQ(hull_single.vertices[0].y, 5.0);

    std::vector<Point2D> collinear{{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    auto hull_collinear = convex_hull_2d_chan(collinear);
    ASSERT_EQ(hull_collinear.vertices.size(), 2u);
    EXPECT_DOUBLE_EQ(hull_collinear.vertices[0].x, 0.0);
    EXPECT_DOUBLE_EQ(hull_collinear.vertices[0].y, 0.0);
    EXPECT_DOUBLE_EQ(hull_collinear.vertices[1].x, 3.0);
    EXPECT_DOUBLE_EQ(hull_collinear.vertices[1].y, 3.0);
}

TEST(ConvexHull2DTest, ChanEquivalenceWithAndrew) {
    // Generate random points inside a unit circle
    std::vector<Point2D> pts;
    const std::size_t num_points = 500;
    pts.reserve(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        double r = std::sqrt(static_cast<double>(i) / num_points);
        double theta = static_cast<double>(i) * 2.399963229728653; // golden angle
        pts.push_back({r * std::cos(theta), r * std::sin(theta)});
    }

    auto hull_andrew = convex_hull_2d_andrew(pts);
    auto hull_chan = convex_hull_2d_chan(pts);

    ASSERT_EQ(hull_chan.vertices.size(), hull_andrew.vertices.size());
    EXPECT_NEAR(hull_chan.area, hull_andrew.area, 1e-10);
    EXPECT_NEAR(hull_chan.perimeter, hull_andrew.perimeter, 1e-10);

    for (std::size_t i = 0; i < hull_andrew.vertices.size(); ++i) {
        EXPECT_NEAR(hull_chan.vertices[i].x, hull_andrew.vertices[i].x, 1e-12);
        EXPECT_NEAR(hull_chan.vertices[i].y, hull_andrew.vertices[i].y, 1e-12);
    }
}

TEST(ConvexHull3DTest, CoplanarPointsDegeneracy) {
    // 8 coplanar points on the z = 5.0 plane
    std::vector<Point3D> pts{
        {0.0, 0.0, 5.0}, {2.0, 0.0, 5.0}, {2.0, 2.0, 5.0}, {0.0, 2.0, 5.0}, {1.0, 0.0, 5.0},
        {2.0, 1.0, 5.0}, {1.0, 2.0, 5.0}, {0.0, 1.0, 5.0}, {1.0, 1.0, 5.0} // Interior point
    };
    auto hull = convex_hull_3d_quickhull(pts);
    // Convex hull of a coplanar square with intermediate and interior points has 4 vertices
    EXPECT_EQ(hull.vertices.size(), 4u);
    EXPECT_DOUBLE_EQ(hull.volume, 0.0);
    EXPECT_NEAR(hull.surface_area, 4.0, 1e-9); // Area of 2x2 square is 4.0
}

// ============================================================================
// 4. 3D Quickhull Polyhedron Generation
// ============================================================================

TEST(ConvexHull3DTest, EmptyAndSinglePoint) {
    std::vector<Point3D> empty;
    auto hull_empty = convex_hull_3d_quickhull(empty);
    EXPECT_TRUE(hull_empty.vertices.empty());
    EXPECT_TRUE(hull_empty.triangle_facets.empty());
    EXPECT_DOUBLE_EQ(hull_empty.volume, 0.0);
    EXPECT_DOUBLE_EQ(hull_empty.surface_area, 0.0);

    std::vector<Point3D> single{{1.0, 2.0, 3.0}};
    auto hull_single = convex_hull_3d_quickhull(single);
    ASSERT_EQ(hull_single.vertices.size(), 1u);
    EXPECT_TRUE(hull_single.triangle_facets.empty());
    EXPECT_DOUBLE_EQ(hull_single.volume, 0.0);
    EXPECT_DOUBLE_EQ(hull_single.surface_area, 0.0);
}

TEST(ConvexHull3DTest, RegularTetrahedron) {
    std::vector<Point3D> pts{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    auto hull = convex_hull_3d_quickhull(pts);
    EXPECT_EQ(hull.vertices.size(), 4u);
    EXPECT_EQ(hull.triangle_facets.size(), 4u);
    EXPECT_NEAR(hull.volume, 1.0 / 6.0, 1e-10);
    EXPECT_NEAR(hull.surface_area, 1.5 + std::sqrt(3.0) / 2.0, 1e-10);
}

TEST(ConvexHull3DTest, UnitCubeWithInteriorPoints) {
    std::vector<Point3D> pts{{-1.0, -1.0, -1.0},
                             {1.0, -1.0, -1.0},
                             {1.0, 1.0, -1.0},
                             {-1.0, 1.0, -1.0},
                             {-1.0, -1.0, 1.0},
                             {1.0, -1.0, 1.0},
                             {1.0, 1.0, 1.0},
                             {-1.0, 1.0, 1.0},
                             // Interior points
                             {0.0, 0.0, 0.0},
                             {0.2, -0.3, 0.5},
                             {-0.4, 0.1, -0.2}};
    auto hull = convex_hull_3d_quickhull(pts);
    EXPECT_EQ(hull.vertices.size(), 8u);
    EXPECT_EQ(hull.triangle_facets.size(), 12u);
    EXPECT_NEAR(hull.volume, 8.0, 1e-9);
    EXPECT_NEAR(hull.surface_area, 24.0, 1e-9);
}
