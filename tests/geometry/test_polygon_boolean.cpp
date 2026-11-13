/**
 * @file test_polygon_boolean.cpp
 * @brief Unit tests for Martinez-Rueda-Feito 2D Polygon Boolean Operations.
 */

#include "algoat/geometry/polygon_boolean.hpp"
#include "algoat/geometry/predicates.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <vector>

using namespace algoat::geometry;

TEST(PolygonBooleanSlice1, PolygonMetadataEmptyAndArea) {
    BooleanPolygon<> empty_poly;
    EXPECT_TRUE(empty_poly.empty());
    EXPECT_DOUBLE_EQ(empty_poly.area(), 0.0);

    // Default PointType is std::array<double, 2>
    BooleanPolygon<> square;
    square.outer_ring = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
    EXPECT_FALSE(square.empty());
    EXPECT_NEAR(square.area(), 100.0, 1e-12);

    // CW outer ring should still report positive net area
    BooleanPolygon<Point2D> cw_with_hole;
    cw_with_hole.outer_ring = {{0.0, 0.0}, {0.0, 10.0}, {10.0, 10.0}, {10.0, 0.0}};
    cw_with_hole.holes = {{{2.0, 2.0}, {8.0, 2.0}, {8.0, 8.0}, {2.0, 8.0}}};
    EXPECT_NEAR(cw_with_hole.area(), 100.0 - 36.0, 1e-12);

    // Subnamespace alias boolean::Polygon<PointType>
    boolean::Polygon<Point2D> aliased;
    aliased.outer_ring = {{0.0, 0.0}, {4.0, 0.0}, {0.0, 3.0}};
    EXPECT_NEAR(aliased.area(), 6.0, 1e-12);
}

TEST(PolygonBooleanSlice1, EmptyInputsAcrossAllOperations) {
    BooleanPolygon<> empty;
    BooleanPolygon<> sq;
    sq.outer_ring = {{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {0.0, 4.0}};

    EXPECT_TRUE(polygon_boolean(empty, sq, BooleanOp::Intersection).empty());
    EXPECT_TRUE(polygon_boolean(sq, empty, BooleanOp::Intersection).empty());
    EXPECT_TRUE(polygon_boolean(empty, sq, BooleanOp::Difference).empty());

    auto diff_res = polygon_boolean(sq, empty, BooleanOp::Difference);
    ASSERT_EQ(diff_res.size(), 1u);
    EXPECT_NEAR(diff_res[0].area(), 16.0, 1e-12);

    auto union_res = polygon_boolean(empty, sq, BooleanOp::Union);
    ASSERT_EQ(union_res.size(), 1u);
    EXPECT_NEAR(union_res[0].area(), 16.0, 1e-12);

    auto xor_res = polygon_boolean(empty, sq, BooleanOp::Xor);
    ASSERT_EQ(xor_res.size(), 1u);
    EXPECT_NEAR(xor_res[0].area(), 16.0, 1e-12);
}

namespace {

template <typename PointType> double total_area(const MultiPolygon<PointType>& mp) {
    double sum = 0.0;
    for (const auto& poly : mp) {
        sum += poly.area();
    }
    return sum;
}

template <typename PointType>
BooleanPolygon<PointType> make_rect(double x0, double y0, double x1, double y1) {
    BooleanPolygon<PointType> poly;
    poly.outer_ring = {
        detail::make_point<PointType>(x0, y0),
        detail::make_point<PointType>(x1, y0),
        detail::make_point<PointType>(x1, y1),
        detail::make_point<PointType>(x0, y1),
    };
    return poly;
}

} // namespace

TEST(PolygonBooleanSlice2, DisjointSquaresAndStrictContainment) {
    auto a = make_rect<std::array<double, 2>>(0.0, 0.0, 4.0, 4.0);
    auto b_disjoint = make_rect<std::array<double, 2>>(10.0, 0.0, 14.0, 4.0);

    EXPECT_TRUE(polygon_intersection(a, b_disjoint).empty());

    auto u_disjoint = polygon_union(a, b_disjoint);
    ASSERT_EQ(u_disjoint.size(), 2u);
    EXPECT_NEAR(total_area(u_disjoint), 32.0, 1e-12);

    auto d_disjoint = polygon_difference(a, b_disjoint);
    ASSERT_EQ(d_disjoint.size(), 1u);
    EXPECT_NEAR(total_area(d_disjoint), 16.0, 1e-12);

    auto x_disjoint = polygon_xor(a, b_disjoint);
    ASSERT_EQ(x_disjoint.size(), 2u);
    EXPECT_NEAR(total_area(x_disjoint), 32.0, 1e-12);

    // Strictly contained inner square [1, 3] x [1, 3] inside [0, 4] x [0, 4]
    auto inner = make_rect<std::array<double, 2>>(1.0, 1.0, 3.0, 3.0);

    auto i_cont = polygon_intersection(a, inner);
    ASSERT_EQ(i_cont.size(), 1u);
    EXPECT_NEAR(total_area(i_cont), 4.0, 1e-12);

    auto u_cont = polygon_union(a, inner);
    ASSERT_EQ(u_cont.size(), 1u);
    EXPECT_TRUE(u_cont[0].holes.empty());
    EXPECT_NEAR(total_area(u_cont), 16.0, 1e-12);

    // A \ inner creates a hole inside A
    auto d_cont = polygon_difference(a, inner);
    ASSERT_EQ(d_cont.size(), 1u);
    ASSERT_EQ(d_cont[0].holes.size(), 1u);
    EXPECT_NEAR(total_area(d_cont), 12.0, 1e-12);

    // inner \ A is empty
    EXPECT_TRUE(polygon_difference(inner, a).empty());

    // A ^ inner also creates a hole inside A
    auto x_cont = polygon_xor(a, inner);
    ASSERT_EQ(x_cont.size(), 1u);
    ASSERT_EQ(x_cont[0].holes.size(), 1u);
    EXPECT_NEAR(total_area(x_cont), 12.0, 1e-12);
}

TEST(PolygonBooleanSlice3, OverlappingSquaresAndConcaveShapes) {
    auto a = make_rect<Point2D>(0.0, 0.0, 10.0, 10.0);
    auto b = make_rect<Point2D>(5.0, 5.0, 15.0, 15.0);

    auto inter = polygon_intersection(a, b);
    ASSERT_EQ(inter.size(), 1u);
    EXPECT_NEAR(total_area(inter), 25.0, 1e-12);

    auto uni = polygon_union(a, b);
    ASSERT_EQ(uni.size(), 1u);
    EXPECT_NEAR(total_area(uni), 175.0, 1e-12);

    auto diff = polygon_difference(a, b);
    ASSERT_EQ(diff.size(), 1u);
    EXPECT_NEAR(total_area(diff), 75.0, 1e-12);

    auto xorr = polygon_xor(a, b);
    // Two touching corner polygons or separated? At (5,5) and (10,10)? No, [0,10]^2 ^ [5,15]^2
    // touches at (5,10) and (10,5)!
    EXPECT_NEAR(total_area(xorr), 150.0, 1e-12);

    // Concave U-shape intersected with horizontal bar splitting into 2 disjoint rectangles
    BooleanPolygon<Point2D> u_shape;
    u_shape.outer_ring = {{0, 0}, {10, 0}, {10, 10}, {7, 10}, {7, 3}, {3, 3}, {3, 10}, {0, 10}};
    auto bar = make_rect<Point2D>(-1.0, 5.0, 11.0, 8.0);
    auto u_inter = polygon_intersection(u_shape, bar);
    ASSERT_EQ(u_inter.size(), 2u);
    EXPECT_NEAR(total_area(u_inter), 18.0, 1e-12);
}

TEST(PolygonBooleanSlice4, IdenticalAdjacentSharedEdgesAndCollinearOverlaps) {
    // 1. Identical polygons
    auto a = make_rect<Point2D>(0.0, 0.0, 4.0, 4.0);
    auto identical = make_rect<Point2D>(0.0, 0.0, 4.0, 4.0);
    EXPECT_NEAR(total_area(polygon_intersection(a, identical)), 16.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_union(a, identical)), 16.0, 1e-12);
    EXPECT_TRUE(polygon_difference(a, identical).empty());
    EXPECT_TRUE(polygon_xor(a, identical).empty());

    // 2. Adjacent squares sharing a full vertical edge at x = 4
    auto right = make_rect<Point2D>(4.0, 0.0, 8.0, 4.0);
    EXPECT_TRUE(polygon_intersection(a, right).empty());
    auto u_adj_v = polygon_union(a, right);
    ASSERT_EQ(u_adj_v.size(), 1u);
    EXPECT_NEAR(total_area(u_adj_v), 32.0, 1e-12);
    auto d_adj_v = polygon_difference(a, right);
    ASSERT_EQ(d_adj_v.size(), 1u);
    EXPECT_NEAR(total_area(d_adj_v), 16.0, 1e-12);
    auto x_adj_v = polygon_xor(a, right);
    ASSERT_EQ(x_adj_v.size(), 1u);
    EXPECT_NEAR(total_area(x_adj_v), 32.0, 1e-12);

    // 3. Adjacent squares sharing a full horizontal edge at y = 4
    auto top = make_rect<Point2D>(0.0, 4.0, 4.0, 8.0);
    EXPECT_TRUE(polygon_intersection(a, top).empty());
    auto u_adj_h = polygon_union(a, top);
    ASSERT_EQ(u_adj_h.size(), 1u);
    EXPECT_NEAR(total_area(u_adj_h), 32.0, 1e-12);

    // 4. Staggered & strictly-contained collinear overlap:
    // Subject [0, 10] x [0, 4], Clip [2, 6] x [4, 7] (touches strictly inside top edge)
    // and Clip2 [2, 12] x [1, 4] (shares partial bottom/top? Here shares right-half bottom edge
    // [2,10] at y=4? No, at y=1..4 it overlaps [2,10]x[1,4])
    auto wide = make_rect<Point2D>(0.0, 0.0, 10.0, 4.0);
    auto cap = make_rect<Point2D>(2.0, 4.0, 6.0, 7.0);
    EXPECT_TRUE(polygon_intersection(wide, cap).empty());
    auto u_cap = polygon_union(wide, cap);
    ASSERT_EQ(u_cap.size(), 1u);
    EXPECT_NEAR(total_area(u_cap), 52.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_difference(wide, cap)), 40.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_difference(cap, wide)), 12.0, 1e-12);

    // Partial staggered overlap with shared collinear top segment [2, 10] at y = 4
    auto overlap_edge = make_rect<Point2D>(2.0, 1.0, 12.0, 4.0);
    // Intersection: [2, 10] x [1, 4] = 8 * 3 = 24
    // Union: 40 + 30 - 24 = 46
    // Diff (wide \ overlap_edge): 40 - 24 = 16
    // Xor: 46 - 24 = 22
    EXPECT_NEAR(total_area(polygon_intersection(wide, overlap_edge)), 24.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_union(wide, overlap_edge)), 46.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_difference(wide, overlap_edge)), 16.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_xor(wide, overlap_edge)), 22.0, 1e-12);
}

TEST(PolygonBooleanSlice4, TJunctionCombAndVerticalCrossings) {
    // Base bar [0, 12] x [0, 2], comb teeth sitting flush on y = 2:
    // Teeth at [1,3]x[2,6], [5,7]x[2,6], [9,11]x[2,6] joined across y=[6,8]
    auto base = make_rect<Point2D>(0.0, 0.0, 12.0, 2.0);
    BooleanPolygon<Point2D> comb;
    comb.outer_ring = {
        {1.0, 2.0}, {3.0, 2.0}, {3.0, 6.0}, {5.0, 6.0},  {5.0, 2.0},  {7.0, 2.0},
        {7.0, 6.0}, {9.0, 6.0}, {9.0, 2.0}, {11.0, 2.0}, {11.0, 8.0}, {1.0, 8.0},
    };
    // Comb area = (10 * 2) + 3 * (2 * 4) = 20 + 24 = 44; Base area = 24.
    // Touching only along y = 2 segments [1,3], [5,7], [9,11]!
    // Union should produce 1 outer ring with 2 enclosed holes ([3,5]x[2,6] and [7,9]x[2,6])!
    EXPECT_TRUE(polygon_intersection(base, comb).empty());
    auto comb_union = polygon_union(base, comb);
    ASSERT_EQ(comb_union.size(), 1u);
    EXPECT_EQ(comb_union[0].holes.size(), 2u);
    EXPECT_NEAR(total_area(comb_union), 68.0, 1e-12);

    // Vertical diamond vs axis-aligned vertical strip (exact vertical-diagonal & vertical-vertical
    // overlaps)
    BooleanPolygon<Point2D> diamond;
    diamond.outer_ring = {{0.0, -4.0}, {4.0, 0.0}, {0.0, 4.0}, {-4.0, 0.0}}; // area = 32
    auto vstrip = make_rect<Point2D>(0.0, -5.0, 2.0, 5.0);                   // area = 20
    // Intersection is right half [0,2] of diamond: trapezoid from y=+-4 at x=0 to y=+-2 at x=2 ->
    // avg height (8+4)/2 * 2 = 12
    EXPECT_NEAR(total_area(polygon_intersection(diamond, vstrip)), 12.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_union(diamond, vstrip)), 40.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_difference(diamond, vstrip)), 20.0, 1e-12);
    EXPECT_NEAR(total_area(polygon_xor(diamond, vstrip)), 28.0, 1e-12);
}

TEST(PolygonBooleanSlice5, NestedIslandsHolesAndPinchedVertices) {
    // 1. Concentric bullseye (Depth 0 -> 1 -> 2 -> 3)
    auto frame_outer = make_rect<Point2D>(0.0, 0.0, 20.0, 20.0);
    frame_outer.holes = {
        {{2.0, 2.0}, {18.0, 2.0}, {18.0, 18.0}, {2.0, 18.0}}}; // area = 400 - 256 = 144

    auto frame_inner = make_rect<Point2D>(4.0, 4.0, 16.0, 16.0);
    frame_inner.holes = {
        {{6.0, 6.0}, {14.0, 6.0}, {14.0, 14.0}, {6.0, 14.0}}}; // area = 144 - 64 = 80

    auto bullseye = polygon_union(frame_outer, frame_inner);
    ASSERT_EQ(bullseye.size(), 2u);
    EXPECT_NEAR(total_area(bullseye), 224.0, 1e-12);
    for (const auto& poly : bullseye) {
        EXPECT_GT(detail::ring_signed_area<Point2D>(poly.outer_ring), 0.0);
        ASSERT_EQ(poly.holes.size(), 1u);
        EXPECT_LT(detail::ring_signed_area<Point2D>(poly.holes[0]), 0.0);
    }

    // Union filling a hole completely and partially
    auto plug_full = make_rect<Point2D>(2.0, 2.0, 18.0, 18.0);
    auto filled = polygon_union(frame_outer, plug_full);
    ASSERT_EQ(filled.size(), 1u);
    EXPECT_TRUE(filled[0].holes.empty());
    EXPECT_NEAR(filled[0].area(), 400.0, 1e-12);

    auto plug_half = make_rect<Point2D>(2.0, 2.0, 10.0, 18.0);
    auto half_filled = polygon_union(frame_outer, plug_half);
    ASSERT_EQ(half_filled.size(), 1u);
    EXPECT_EQ(half_filled[0].holes.size(), 1u);
    EXPECT_NEAR(half_filled[0].area(), 272.0, 1e-12);

    // 2. Hole touching outer boundary at a corner vertex & shared edge
    BooleanPolygon<Point2D> poly_touch;
    poly_touch.outer_ring = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};
    // Square cut touching (0, 0) corner: [0, 4] x [0, 4] via difference -> L-shape (no hole)
    auto corner_cut = make_rect<Point2D>(0.0, 0.0, 4.0, 4.0);
    auto l_diff = polygon_difference(poly_touch, corner_cut);
    ASSERT_EQ(l_diff.size(), 1u);
    EXPECT_TRUE(l_diff[0].holes.empty());
    EXPECT_NEAR(total_area(l_diff), 84.0, 1e-12);

    // Diamond hole touching outer square at 4 midpoints: (5,0), (10,5), (5,10), (0,5)
    // Wait: (0,0)-(5,0)-(0,5) etc splits square into 4 separate corner triangles!
    BooleanPolygon<Point2D> diamond_hole;
    diamond_hole.outer_ring = {{5.0, 0.0}, {10.0, 5.0}, {5.0, 10.0}, {0.0, 5.0}}; // area = 50
    auto four_corners = polygon_difference(poly_touch, diamond_hole);
    ASSERT_EQ(four_corners.size(), 4u);
    EXPECT_NEAR(total_area(four_corners), 50.0, 1e-12);

    // Checkerboard touching-corner union: [0,4]x[0,4] and [4,8]x[4,8]
    auto c1 = make_rect<Point2D>(0.0, 0.0, 4.0, 4.0);
    auto c2 = make_rect<Point2D>(4.0, 4.0, 8.0, 8.0);
    auto chk = polygon_union(c1, c2);
    ASSERT_EQ(chk.size(), 2u);
    EXPECT_NEAR(total_area(chk), 32.0, 1e-12);

    // Two interior holes touching at a shared vertex (pinched figure-8 hole)
    auto big = make_rect<Point2D>(0.0, 0.0, 10.0, 10.0);
    big.holes = {
        {{2.0, 2.0}, {5.0, 2.0}, {5.0, 5.0}, {2.0, 5.0}},
        {{5.0, 5.0}, {8.0, 5.0}, {8.0, 8.0}, {5.0, 8.0}},
    };
    auto big_self = polygon_union(big, BooleanPolygon<Point2D>{});
    ASSERT_EQ(big_self.size(), 1u);
    EXPECT_NEAR(total_area(big_self), 100.0 - 9.0 - 9.0, 1e-12);
}

TEST(PolygonBooleanSlice5, SelfIntersectionsSpikesRotationsAndLargeOffsets) {
    // 1. Bowtie (self-intersecting) vs empty / vs square
    BooleanPolygon<Point2D> bowtie;
    bowtie.outer_ring = {{0.0, 0.0}, {4.0, 4.0}, {4.0, 0.0}, {0.0, 4.0}}; // 2 triangles each area 4
    auto bow_norm = polygon_union(bowtie, BooleanPolygon<Point2D>{});
    ASSERT_EQ(bow_norm.size(), 2u);
    EXPECT_NEAR(total_area(bow_norm), 8.0, 1e-12);

    auto right_box = make_rect<Point2D>(2.0, 0.0, 5.0, 4.0);
    auto bow_inter = polygon_intersection(bowtie, right_box);
    EXPECT_NEAR(total_area(bow_inter), 4.0, 1e-12);

    // 2. 180-degree backtracking spike on square edge
    BooleanPolygon<Point2D> spiked;
    spiked.outer_ring = {
        {0.0, 0.0}, {5.0, 0.0}, {5.0, 3.0}, {5.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0},
    };
    auto other_sq = make_rect<Point2D>(2.0, 2.0, 8.0, 8.0);
    EXPECT_NEAR(total_area(polygon_intersection(spiked, other_sq)), 36.0, 1e-12);

    // 3. Rotated unit squares (15 deg and 45 deg) + exact inclusion-exclusion
    auto rotate_rect = [](double w, double h, double deg, double ox, double oy) {
        const double rad = deg * std::numbers::pi / 180.0;
        const double c = std::cos(rad);
        const double s = std::sin(rad);
        std::array<Point2D, 4> corners{{{-w, -h}, {w, -h}, {w, h}, {-w, h}}};
        BooleanPolygon<Point2D> p;
        for (const auto& pt : corners) {
            p.outer_ring.push_back({ox + pt.x * c - pt.y * s, oy + pt.x * s + pt.y * c});
        }
        return p;
    };
    auto r15 = rotate_rect(5.0, 5.0, 15.0, 0.0, 0.0);
    auto r45 = rotate_rect(5.0, 5.0, 45.0, 1.5, -0.5);
    const double ai = total_area(polygon_intersection(r15, r45));
    const double au = total_area(polygon_union(r15, r45));
    const double ad1 = total_area(polygon_difference(r15, r45));
    const double ad2 = total_area(polygon_difference(r45, r15));
    const double ax = total_area(polygon_xor(r15, r45));
    EXPECT_NEAR(au + ai, r15.area() + r45.area(), 1e-10);
    EXPECT_NEAR(ad1 + ai, r15.area(), 1e-10);
    EXPECT_NEAR(ad2 + ai, r45.area(), 1e-10);
    EXPECT_NEAR(ax, au - ai, 1e-10);

    // 4. Large coordinate offset (~1e9, typical UTM/ECEF meters)
    constexpr double kBig = 1.0e9;
    auto far_a = make_rect<Point2D>(kBig + 0.0, kBig + 0.0, kBig + 10.0, kBig + 10.0);
    auto far_b = make_rect<Point2D>(kBig + 4.0, kBig + 4.0, kBig + 14.0, kBig + 14.0);
    EXPECT_NEAR(total_area(polygon_intersection(far_a, far_b)), 36.0, 1e-9);
    EXPECT_NEAR(total_area(polygon_union(far_a, far_b)), 164.0, 1e-9);
    EXPECT_NEAR(total_area(polygon_difference(far_a, far_b)), 64.0, 1e-9);
    EXPECT_NEAR(total_area(polygon_xor(far_a, far_b)), 128.0, 1e-9);

    // 5. Hole sharing an edge with outer boundary -> opens up into C-shape (0 holes)
    auto c_frame = make_rect<Point2D>(0.0, 0.0, 10.0, 10.0);
    c_frame.holes = {{{0.0, 2.0}, {4.0, 2.0}, {4.0, 8.0}, {0.0, 8.0}}};
    auto c_norm = polygon_union(c_frame, BooleanPolygon<Point2D>{});
    ASSERT_EQ(c_norm.size(), 1u);
    EXPECT_TRUE(c_norm[0].holes.empty());
    EXPECT_NEAR(total_area(c_norm), 76.0, 1e-12);

    // 6. Two interior holes sharing an edge -> merge into 1 hole
    auto two_holes = make_rect<Point2D>(0.0, 0.0, 10.0, 10.0);
    two_holes.holes = {
        {{2.0, 2.0}, {5.0, 2.0}, {5.0, 8.0}, {2.0, 8.0}},
        {{5.0, 2.0}, {8.0, 2.0}, {8.0, 8.0}, {5.0, 8.0}},
    };
    auto merged_holes = polygon_union(two_holes, BooleanPolygon<Point2D>{});
    ASSERT_EQ(merged_holes.size(), 1u);
    ASSERT_EQ(merged_holes[0].holes.size(), 1u);
    EXPECT_NEAR(total_area(merged_holes), 64.0, 1e-12);

    // 7. Sliver triangle near-collinear predicate stress
    BooleanPolygon<Point2D> sliver;
    sliver.outer_ring = {{0.0, 0.0}, {10.0, 0.0}, {5.0, 1e-9}};
    auto box = make_rect<Point2D>(2.0, -1.0, 8.0, 1.0);
    const double sliver_inter = total_area(polygon_intersection(sliver, box));
    const double sliver_union = total_area(polygon_union(sliver, box));
    EXPECT_NEAR(sliver_union + sliver_inter, sliver.area() + box.area(), 1e-12);

    // 8. Fuzzer regression: intersecting hole + cross-diagonal triangles (AADDtsEPRXYAAEF66e/v)
    BooleanPolygon<Point2D> fuz_a;
    fuz_a.outer_ring = {{1.0, 12.0}, {-1.0, -13.0}, {3.0, -16.0}};
    fuz_a.holes = {{{-2.0, -2.0}, {2.0, -2.0}, {2.0, 2.0}, {-2.0, 2.0}}};
    BooleanPolygon<Point2D> fuz_b;
    fuz_b.outer_ring = {{-16.0, 16.0}, {7.0, -14.0}, {-8.0, -8.0}};
    const double fa = total_area(polygon_union(fuz_a, BooleanPolygon<Point2D>{}));
    const double fb = total_area(polygon_union(fuz_b, BooleanPolygon<Point2D>{}));
    const double fi = total_area(polygon_intersection(fuz_a, fuz_b));
    const double fu = total_area(polygon_union(fuz_a, fuz_b));
    EXPECT_NEAR(fu + fi, fa + fb, 1e-9);

    // 9. Non-finite coordinates (NaN / Inf) safely ignored without UB or crashes
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const double inf_val = std::numeric_limits<double>::infinity();
    BooleanPolygon<Point2D> nan_poly;
    nan_poly.outer_ring = {{0.0, 0.0}, {nan_val, 0.0}, {10.0, 10.0}};
    EXPECT_TRUE(polygon_intersection(nan_poly, other_sq).empty());
    EXPECT_NEAR(total_area(polygon_union(nan_poly, other_sq)), 36.0, 1e-12);

    BooleanPolygon<Point2D> inf_poly;
    inf_poly.outer_ring = {{0.0, 0.0}, {inf_val, 0.0}, {10.0, 10.0}};
    EXPECT_TRUE(polygon_intersection(inf_poly, other_sq).empty());
    EXPECT_NEAR(total_area(polygon_difference(other_sq, inf_poly)), 36.0, 1e-12);
}
