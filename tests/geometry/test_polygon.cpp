#include "algoat/geometry/polygon.hpp"
#include "algoat/geometry/predicates.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::geometry;

// ============================================================================
// 1. Construction & Metadata Accessors
// ============================================================================

TEST(PolygonTest, DefaultConstruction) {
    Polygon poly;
    EXPECT_EQ(poly.edge_count(), 0u);
    EXPECT_FALSE(poly.bounding_box().is_valid());
}

TEST(PolygonTest, DegeneratePolygons) {
    std::vector<Point2D> empty;
    Polygon poly_empty(empty);
    EXPECT_EQ(poly_empty.edge_count(), 0u);
    EXPECT_EQ(poly_empty.contains({0.0, 0.0}), PolygonInclusion::Outside);

    std::vector<Point2D> single = {{1.0, 2.0}};
    Polygon poly_single(single);
    EXPECT_EQ(poly_single.edge_count(), 0u);
    EXPECT_EQ(poly_single.contains({1.0, 2.0}), PolygonInclusion::Outside);

    std::vector<Point2D> line = {{0.0, 0.0}, {2.0, 2.0}};
    Polygon poly_line(line);
    EXPECT_EQ(poly_line.edge_count(), 0u);
    EXPECT_EQ(poly_line.contains({1.0, 1.0}), PolygonInclusion::Outside);
}

TEST(PolygonTest, BasicSquareConstructionAndBoundingBox) {
    // CCW Unit square [0, 10] x [0, 10]
    std::vector<Point2D> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
    Polygon poly(square);
    EXPECT_EQ(poly.edge_count(), 4u);

    const auto& bbox = poly.bounding_box();
    EXPECT_TRUE(bbox.is_valid());
    EXPECT_FLOAT_EQ(bbox.min_x, 0.0f);
    EXPECT_FLOAT_EQ(bbox.min_y, 0.0f);
    EXPECT_FLOAT_EQ(bbox.max_x, 10.0f);
    EXPECT_FLOAT_EQ(bbox.max_y, 10.0f);
}

TEST(PolygonTest, ExplicitlyClosedSquare) {
    // Already closed with duplicate first vertex
    std::vector<Point2D> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}, {0.0, 0.0}};
    Polygon poly(square);
    EXPECT_EQ(poly.edge_count(), 4u);
}

// ============================================================================
// 2. Square Containment & Boundary Tests
// ============================================================================

TEST(PolygonTest, SquarePointContainment) {
    std::vector<Point2D> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
    Polygon poly(square);

    // Strictly inside
    EXPECT_EQ(poly.contains({5.0, 5.0}), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({1.0, 1.0}), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({9.0, 9.0}), PolygonInclusion::Inside);

    // Strictly outside (outside AABB)
    EXPECT_EQ(poly.contains({-1.0, 5.0}), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({11.0, 5.0}), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({5.0, -1.0}), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({5.0, 11.0}), PolygonInclusion::Outside);

    // Strictly outside (inside AABB range along one axis, but outside polygon)
    EXPECT_EQ(poly.contains({-0.1, 5.0}), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({10.1, 5.0}), PolygonInclusion::Outside);

    // Vertices -> OnBoundary
    EXPECT_EQ(poly.contains({0.0, 0.0}), PolygonInclusion::OnBoundary);
    EXPECT_EQ(poly.contains({10.0, 0.0}), PolygonInclusion::OnBoundary);
    EXPECT_EQ(poly.contains({10.0, 10.0}), PolygonInclusion::OnBoundary);
    EXPECT_EQ(poly.contains({0.0, 10.0}), PolygonInclusion::OnBoundary);

    // Edges -> OnBoundary
    EXPECT_EQ(poly.contains({5.0, 0.0}), PolygonInclusion::OnBoundary);
    EXPECT_EQ(poly.contains({10.0, 5.0}), PolygonInclusion::OnBoundary);
    EXPECT_EQ(poly.contains({5.0, 10.0}), PolygonInclusion::OnBoundary);
    EXPECT_EQ(poly.contains({0.0, 5.0}), PolygonInclusion::OnBoundary);
}

TEST(PolygonTest, BothInclusionRulesOnSimpleConvex) {
    std::vector<Point2D> triangle = {{0.0, 0.0}, {10.0, 0.0}, {0.0, 10.0}};
    Polygon poly(triangle);

    // EvenOdd and NonZeroWinding agree on simple convex geometry
    for (auto rule : {InclusionRule::EvenOdd, InclusionRule::NonZeroWinding}) {
        EXPECT_EQ(poly.contains({2.0, 2.0}, rule), PolygonInclusion::Inside);
        EXPECT_EQ(poly.contains({8.0, 8.0}, rule), PolygonInclusion::Outside);
        EXPECT_EQ(poly.contains({0.0, 0.0}, rule), PolygonInclusion::OnBoundary);
        EXPECT_EQ(poly.contains({5.0, 0.0}, rule), PolygonInclusion::OnBoundary);
        EXPECT_EQ(poly.contains({5.0, 5.0}, rule), PolygonInclusion::OnBoundary);
    }
}

// ============================================================================
// 3. Concave & Ray-Vertex Alignment Edge Cases
// ============================================================================

TEST(PolygonTest, ConcaveLShapedPolygon) {
    // L-shape in [0, 20] x [0, 20]
    // Vertices: (0,0) -> (20,0) -> (20,10) -> (10,10) -> (10,20) -> (0,20)
    std::vector<Point2D> l_shape = {{0.0, 0.0},   {20.0, 0.0},  {20.0, 10.0},
                                    {10.0, 10.0}, {10.0, 20.0}, {0.0, 20.0}};
    Polygon poly(l_shape);

    EXPECT_EQ(poly.contains({5.0, 5.0}), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({15.0, 5.0}), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({5.0, 15.0}), PolygonInclusion::Inside);

    // The cutout notch [10, 20] x [10, 20] is strictly outside
    EXPECT_EQ(poly.contains({15.0, 15.0}), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({12.0, 12.0}), PolygonInclusion::Outside);

    // Ray from (-5, 10) directly hits the reflex corner (10, 10) and continues
    EXPECT_EQ(poly.contains({-5.0, 10.0}), PolygonInclusion::Outside);

    // Reflex corner is OnBoundary
    EXPECT_EQ(poly.contains({10.0, 10.0}), PolygonInclusion::OnBoundary);
}

TEST(PolygonTest, RayPassingThroughVertexTangentAndCrossing) {
    // Arrow-shaped polygon with a peak:
    // (0,0) -> (10, 5) -> (0, 10) -> (3, 5)
    std::vector<Point2D> arrow = {{0.0, 0.0}, {10.0, 5.0}, {0.0, 10.0}, {3.0, 5.0}};
    Polygon poly(arrow);

    // Point in indent behind notch (3, 5) is outside
    EXPECT_EQ(poly.contains({1.0, 5.0}), PolygonInclusion::Outside);
    // Point inside arrow body is inside
    EXPECT_EQ(poly.contains({5.0, 5.0}), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({12.0, 5.0}), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({-2.0, 5.0}), PolygonInclusion::Outside);

    // On internal vertex (3, 5)
    EXPECT_EQ(poly.contains({3.0, 5.0}), PolygonInclusion::OnBoundary);
    // On tip vertex (10, 5)
    EXPECT_EQ(poly.contains({10.0, 5.0}), PolygonInclusion::OnBoundary);
}

// ============================================================================
// 4. Complex Non-Simple Polygons (Self-Intersections & Holes)
// ============================================================================

TEST(PolygonTest, FigureEightSelfIntersecting) {
    // Figure-of-eight polygon intersecting at (5, 5):
    // Left loop (0,0) -> (5,5) -> (0,10)
    // Right loop (10,10) -> (5,5) -> (10,0)
    std::vector<Point2D> fig8 = {{0.0, 0.0},   {5.0, 5.0}, {0.0, 10.0},
                                 {10.0, 10.0}, {5.0, 5.0}, {10.0, 0.0}};
    Polygon poly(fig8);

    // Under NonZeroWinding, both lobes (top [5, 10] and bottom [0, 5]) are inside
    EXPECT_EQ(poly.contains({5.0, 8.0}, InclusionRule::NonZeroWinding), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({5.0, 2.0}, InclusionRule::NonZeroWinding), PolygonInclusion::Inside);

    // Lateral cutouts at y=5 outside the lobes
    EXPECT_EQ(poly.contains({2.0, 5.0}, InclusionRule::NonZeroWinding), PolygonInclusion::Outside);
    EXPECT_EQ(poly.contains({8.0, 5.0}, InclusionRule::NonZeroWinding), PolygonInclusion::Outside);

    // Intersection point is OnBoundary
    EXPECT_EQ(poly.contains({5.0, 5.0}, InclusionRule::NonZeroWinding),
              PolygonInclusion::OnBoundary);
}

TEST(PolygonTest, StarPentagramWindingVsEvenOdd) {
    // Regular 5-pointed star polygon
    // In EvenOdd, center is a hole (2 crossings -> outside)
    // In NonZeroWinding, center is double-wrapped (wn = 2 -> inside)
    std::vector<Point2D> star = {{0.0, 10.0},   {2.35, 2.65}, {9.51, 3.09},   {3.69, -1.09},
                                 {5.88, -8.09}, {0.0, -3.82}, {-5.88, -8.09}, {-3.69, -1.09},
                                 {-9.51, 3.09}, {-2.35, 2.65}};
    Polygon poly(star);

    // Spike tips are inside for both rules
    EXPECT_EQ(poly.contains({0.0, 7.0}, InclusionRule::EvenOdd), PolygonInclusion::Inside);
    EXPECT_EQ(poly.contains({0.0, 7.0}, InclusionRule::NonZeroWinding), PolygonInclusion::Inside);

    // Center point (0, 0)
    EXPECT_EQ(poly.contains({0.0, 0.0}, InclusionRule::NonZeroWinding), PolygonInclusion::Inside);
}

// ============================================================================
// 5. Batch Queries & SIMD Equivalence
// ============================================================================

TEST(PolygonTest, BatchContainsMatchesSingleContains) {
    std::vector<Point2D> square = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
    Polygon poly(square);

    std::vector<Point2D> queries = {
        {5.0, 5.0},   // Inside
        {-2.0, 5.0},  // Outside left
        {12.0, 5.0},  // Outside right
        {5.0, -2.0},  // Outside bottom
        {5.0, 12.0},  // Outside top
        {0.0, 0.0},   // OnBoundary vertex
        {10.0, 10.0}, // OnBoundary vertex
        {5.0, 0.0},   // OnBoundary edge
        {10.0, 5.0},  // OnBoundary edge
        {1.0, 1.0},   // Inside
        {9.0, 9.0},   // Inside
        {15.0, 15.0}, // Outside AABB
        {0.0, 5.0},   // OnBoundary edge
        {5.0, 10.0},  // OnBoundary edge
        {0.0, 10.0},  // OnBoundary vertex
        {10.0, 0.0}   // OnBoundary vertex
    };

    std::vector<PolygonInclusion> results_batch(queries.size());

    for (auto rule : {InclusionRule::NonZeroWinding, InclusionRule::EvenOdd}) {
        poly.batch_contains(queries, results_batch, rule);

        for (std::size_t i = 0; i < queries.size(); ++i) {
            PolygonInclusion expected = poly.contains(queries[i], rule);
            EXPECT_EQ(results_batch[i], expected)
                << "Mismatch at query index " << i << " (" << queries[i].x << ", " << queries[i].y
                << ") with rule " << static_cast<int>(rule);
        }
    }
}

TEST(PolygonTest, BatchContainsArbitrarySize) {
    // Tests sizes that do not divide SIMD vector width evenly: 1, 3, 5, 7, 13, 31, 65
    std::vector<Point2D> poly_pts = {
        {0.0, 0.0}, {100.0, 0.0}, {100.0, 100.0}, {50.0, 50.0}, {0.0, 100.0}};
    Polygon poly(poly_pts);

    for (std::size_t count : {1u, 2u, 3u, 5u, 7u, 13u, 31u, 65u, 128u}) {
        std::vector<Point2D> queries;
        queries.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            double val = static_cast<double>(i) * 1.5;
            queries.push_back({val, val});
        }

        std::vector<PolygonInclusion> results(count);
        poly.batch_contains(queries, results, InclusionRule::NonZeroWinding);

        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_EQ(results[i], poly.contains(queries[i], InclusionRule::NonZeroWinding))
                << "Count: " << count << " at index " << i;
        }
    }
}
