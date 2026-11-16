/**
 * @file test_smallest_enclosing_ball.cpp
 * @brief Unit tests for Welzl's smallest enclosing sphere/circle algorithm.
 */

#include "algoat/geometry/smallest_enclosing_ball.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#if defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__)
#if defined(__has_feature)
#if !__has_feature(address_sanitizer)
#define ALGOAT_TEST_CHECK_MALLINFO2 1
#endif
#else
#define ALGOAT_TEST_CHECK_MALLINFO2 1
#endif
#endif

#if defined(ALGOAT_TEST_CHECK_MALLINFO2)
#include <malloc.h>
#endif

namespace algoat::geometry::testing {

TEST(SmallestEnclosingBallSlice1, BoundingSphereContainsAndVolume2DAnd3D) {
    BoundingSphere<double, 2> circle{{1.0, 2.0}, 5.0};
    EXPECT_TRUE(circle.contains({1.0, 2.0}));
    EXPECT_TRUE(circle.contains({6.0, 2.0}));
    EXPECT_TRUE(circle.contains({4.0, 6.0}));
    EXPECT_TRUE(circle.contains({6.0 + 5e-10, 2.0}, 1e-9));
    EXPECT_FALSE(circle.contains({6.0 + 1e-7, 2.0}, 1e-9));
    EXPECT_FALSE(circle.contains({std::numeric_limits<double>::quiet_NaN(), 2.0}));
    EXPECT_NEAR(circle.volume(), std::numbers::pi * 25.0, 1e-12);

    BoundingSphere<double, 3> sphere{{0.0, -1.0, 2.0}, 3.0};
    EXPECT_TRUE(sphere.contains({0.0, -1.0, 2.0}));
    EXPECT_TRUE(sphere.contains({0.0, 2.0, 2.0}));
    EXPECT_TRUE(sphere.contains({1.0, 1.0, 4.0})); // dist = sqrt(1 + 4 + 4) = 3
    EXPECT_FALSE(sphere.contains({0.0, 2.0 + 1e-6, 2.0}, 1e-9));
    EXPECT_NEAR(sphere.volume(), (4.0 / 3.0) * std::numbers::pi * 27.0, 1e-12);

    BoundingSphere<float, 3> sphere_f{{0.0f, 0.0f, 0.0f}, 2.0f};
    EXPECT_TRUE(sphere_f.contains({2.0f, 0.0f, 0.0f}, 1e-6f));
    EXPECT_FALSE(sphere_f.contains({2.1f, 0.0f, 0.0f}, 1e-6f));
    EXPECT_NEAR(sphere_f.volume(), static_cast<float>((4.0 / 3.0) * std::numbers::pi * 8.0), 1e-5f);

    BoundingSphere<double, 2> zero_r{{3.0, 4.0}, 0.0};
    EXPECT_TRUE(zero_r.contains({3.0, 4.0}));
    EXPECT_FALSE(zero_r.contains({3.0 + 1e-6, 4.0}, 1e-9));
    EXPECT_DOUBLE_EQ(zero_r.volume(), 0.0);

    BoundingSphere<double, 2> neg_r{{0.0, 0.0}, -1.0};
    EXPECT_FALSE(neg_r.contains({0.0, 0.0}));
    EXPECT_DOUBLE_EQ(neg_r.volume(), 0.0);
}

TEST(SmallestEnclosingBallSlice2, ExactSmallAndDegenerateConfigurations2DAnd3D) {
    // Empty inputs
    std::vector<std::array<double, 2>> empty2;
    const auto res_empty2 =
        smallest_enclosing_circle(std::span<const std::array<double, 2>>{empty2});
    EXPECT_DOUBLE_EQ(res_empty2.radius, 0.0);
    EXPECT_DOUBLE_EQ(res_empty2.center[0], 0.0);
    EXPECT_DOUBLE_EQ(res_empty2.center[1], 0.0);

    std::vector<std::array<double, 3>> empty3;
    const auto res_empty3 = smallest_enclosing_ball(std::span<const std::array<double, 3>>{empty3});
    EXPECT_DOUBLE_EQ(res_empty3.radius, 0.0);

    // Single point
    std::vector<std::array<double, 2>> single2 = {{3.5, -7.25}};
    const auto res_s2 = smallest_enclosing_circle(std::span<const std::array<double, 2>>{single2});
    EXPECT_DOUBLE_EQ(res_s2.center[0], 3.5);
    EXPECT_DOUBLE_EQ(res_s2.center[1], -7.25);
    EXPECT_DOUBLE_EQ(res_s2.radius, 0.0);
    EXPECT_TRUE(res_s2.contains(single2[0]));

    std::vector<std::array<double, 3>> single3 = {{-1.0, 2.0, 4.5}};
    const auto res_s3 = smallest_enclosing_ball(std::span<const std::array<double, 3>>{single3});
    EXPECT_DOUBLE_EQ(res_s3.center[0], -1.0);
    EXPECT_DOUBLE_EQ(res_s3.center[1], 2.0);
    EXPECT_DOUBLE_EQ(res_s3.center[2], 4.5);
    EXPECT_DOUBLE_EQ(res_s3.radius, 0.0);

    // Duplicate identical points
    std::vector<std::array<double, 3>> dup3(16, {2.0, -3.0, 5.0});
    const auto res_dup = smallest_enclosing_ball(std::span<const std::array<double, 3>>{dup3});
    EXPECT_DOUBLE_EQ(res_dup.radius, 0.0);
    EXPECT_DOUBLE_EQ(res_dup.center[0], 2.0);
    EXPECT_DOUBLE_EQ(res_dup.center[1], -3.0);
    EXPECT_DOUBLE_EQ(res_dup.center[2], 5.0);

    // Two points -> diametric sphere
    std::vector<std::array<double, 2>> two2 = {{0.0, 0.0}, {6.0, 8.0}};
    const auto res_two2 = smallest_enclosing_circle(std::span<const std::array<double, 2>>{two2});
    EXPECT_NEAR(res_two2.center[0], 3.0, 1e-12);
    EXPECT_NEAR(res_two2.center[1], 4.0, 1e-12);
    EXPECT_NEAR(res_two2.radius, 5.0, 1e-12);

    // 2D right triangle -> circumcircle of hypotenuse
    std::vector<std::array<double, 2>> right2 = {{0.0, 0.0}, {6.0, 0.0}, {0.0, 8.0}};
    const auto res_right2 =
        smallest_enclosing_circle(std::span<const std::array<double, 2>>{right2});
    EXPECT_NEAR(res_right2.center[0], 3.0, 1e-12);
    EXPECT_NEAR(res_right2.center[1], 4.0, 1e-12);
    EXPECT_NEAR(res_right2.radius, 5.0, 1e-12);

    // 2D acute equilateral-like triangle: (0,0), (4,0), (2,3) -> circumcenter (2, 1/6*?)
    // Wait: |(2, y) - (0,0)|^2 = 4 + y^2; |(2,y) - (2,3)|^2 = (3-y)^2 = 9 - 6y + y^2
    // 4 + y^2 = 9 - 6y + y^2 => 6y = 5 => y = 5/6, r = sqrt(4 + 25/36) = 13/6.
    std::vector<std::array<double, 2>> acute2 = {{0.0, 0.0}, {4.0, 0.0}, {2.0, 3.0}};
    const auto res_acute2 =
        smallest_enclosing_circle(std::span<const std::array<double, 2>>{acute2});
    EXPECT_NEAR(res_acute2.center[0], 2.0, 1e-12);
    EXPECT_NEAR(res_acute2.center[1], 5.0 / 6.0, 1e-12);
    EXPECT_NEAR(res_acute2.radius, 13.0 / 6.0, 1e-12);

    // 2D obtuse triangle -> longest edge diameter ball, NOT larger circumcircle!
    // (-5, 0), (5, 0), (0, 0.1) -> circumradius ~ 125.05, SEB radius is 5.0 at (0, 0)!
    std::vector<std::array<double, 2>> obtuse2 = {{0.0, 0.1}, {-5.0, 0.0}, {5.0, 0.0}};
    const auto res_obtuse2 =
        smallest_enclosing_circle(std::span<const std::array<double, 2>>{obtuse2});
    EXPECT_NEAR(res_obtuse2.center[0], 0.0, 1e-12);
    EXPECT_NEAR(res_obtuse2.center[1], 0.0, 1e-12);
    EXPECT_NEAR(res_obtuse2.radius, 5.0, 1e-12);
    for (const auto& p : obtuse2) {
        EXPECT_TRUE(res_obtuse2.contains(p, 1e-9));
    }

    // 2D collinear points -> diametric circle of endpoints
    std::vector<std::array<double, 2>> col2 = {{1.0, 2.0}, {3.0, 6.0}, {5.0, 10.0}, {2.0, 4.0}};
    const auto res_col2 = smallest_enclosing_circle(std::span<const std::array<double, 2>>{col2});
    EXPECT_NEAR(res_col2.center[0], 3.0, 1e-12);
    EXPECT_NEAR(res_col2.center[1], 6.0, 1e-12);
    EXPECT_NEAR(res_col2.radius, std::hypot(2.0, 4.0), 1e-12);

    // 3D acute triangle embedded in 3D space (rotated/non-axis-aligned)
    // (1, 1, 1), (5, 1, 1), (3, 4, 1) -> circumcenter (3, 1 + 5/6, 1), radius 13/6
    std::vector<std::array<double, 3>> acute3_tri = {
        {1.0, 1.0, 1.0}, {5.0, 1.0, 1.0}, {3.0, 4.0, 1.0}};
    const auto res_a3_tri =
        smallest_enclosing_ball(std::span<const std::array<double, 3>>{acute3_tri});
    EXPECT_NEAR(res_a3_tri.center[0], 3.0, 1e-12);
    EXPECT_NEAR(res_a3_tri.center[1], 1.0 + 5.0 / 6.0, 1e-12);
    EXPECT_NEAR(res_a3_tri.center[2], 1.0, 1e-12);
    EXPECT_NEAR(res_a3_tri.radius, 13.0 / 6.0, 1e-12);

    // 3D obtuse triangle -> longest edge diameter ball
    std::vector<std::array<double, 3>> obtuse3_tri = {
        {0.0, 0.2, 0.2}, {-5.0, 0.0, 0.0}, {5.0, 0.0, 0.0}};
    const auto res_o3_tri =
        smallest_enclosing_ball(std::span<const std::array<double, 3>>{obtuse3_tri});
    EXPECT_NEAR(res_o3_tri.center[0], 0.0, 1e-12);
    EXPECT_NEAR(res_o3_tri.center[1], 0.0, 1e-12);
    EXPECT_NEAR(res_o3_tri.center[2], 0.0, 1e-12);
    EXPECT_NEAR(res_o3_tri.radius, 5.0, 1e-12);

    // 3D regular tetrahedron inscribed on sphere of radius sqrt(3) centered at (1, 2, 3)
    // Vertices: (1+1,2+1,3+1), (1+1,2-1,3-1), (1-1,2+1,3-1), (1-1,2-1,3+1)
    std::vector<std::array<double, 3>> reg_tet = {
        {2.0, 3.0, 4.0}, {2.0, 1.0, 2.0}, {0.0, 3.0, 2.0}, {0.0, 1.0, 4.0}};
    const auto res_tet = smallest_enclosing_ball(std::span<const std::array<double, 3>>{reg_tet});
    EXPECT_NEAR(res_tet.center[0], 1.0, 1e-12);
    EXPECT_NEAR(res_tet.center[1], 2.0, 1e-12);
    EXPECT_NEAR(res_tet.center[2], 3.0, 1e-12);
    EXPECT_NEAR(res_tet.radius, std::sqrt(3.0), 1e-12);

    // 3D flat/obtuse tetrahedron (4th point slightly lifted inside 3-point equilateral base sphere)
    // Triangle (1,0,0), (-0.5, sqrt(3)/2, 0), (-0.5, -sqrt(3)/2, 0) has SEB center (0,0,0), r=1.
    // 4th point (0, 0, 0.2) is strictly inside -> circumsphere is HUGE, SEB must have center
    // (0,0,0), r=1!
    const double s32 = std::sqrt(3.0) * 0.5;
    std::vector<std::array<double, 3>> flat_tet = {
        {0.0, 0.0, 0.2}, {1.0, 0.0, 0.0}, {-0.5, s32, 0.0}, {-0.5, -s32, 0.0}};
    const auto res_ft = smallest_enclosing_ball(std::span<const std::array<double, 3>>{flat_tet});
    EXPECT_NEAR(res_ft.center[0], 0.0, 1e-12);
    EXPECT_NEAR(res_ft.center[1], 0.0, 1e-12);
    EXPECT_NEAR(res_ft.center[2], 0.0, 1e-12);
    EXPECT_NEAR(res_ft.radius, 1.0, 1e-12);

    // 3D coplanar square -> radius sqrt(2)
    std::vector<std::array<double, 3>> coplanar4 = {
        {0.0, 0.0, 5.0}, {2.0, 0.0, 5.0}, {2.0, 2.0, 5.0}, {0.0, 2.0, 5.0}};
    const auto res_cop = smallest_enclosing_ball(std::span<const std::array<double, 3>>{coplanar4});
    EXPECT_NEAR(res_cop.center[0], 1.0, 1e-12);
    EXPECT_NEAR(res_cop.center[1], 1.0, 1e-12);
    EXPECT_NEAR(res_cop.center[2], 5.0, 1e-12);
    EXPECT_NEAR(res_cop.radius, std::sqrt(2.0), 1e-12);

    // 100 cocircular points in 2D around (4, -3) with radius 7
    std::vector<std::array<double, 2>> circle100;
    for (int i = 0; i < 100; ++i) {
        const double theta = (2.0 * std::numbers::pi * i) / 100.0;
        circle100.push_back({4.0 + 7.0 * std::cos(theta), -3.0 + 7.0 * std::sin(theta)});
    }
    const auto res_c100 =
        smallest_enclosing_circle(std::span<const std::array<double, 2>>{circle100});
    EXPECT_NEAR(res_c100.center[0], 4.0, 1e-9);
    EXPECT_NEAR(res_c100.center[1], -3.0, 1e-9);
    EXPECT_NEAR(res_c100.radius, 7.0, 1e-9);
    for (const auto& p : circle100) {
        EXPECT_TRUE(res_c100.contains(p, 1e-9));
    }

    // 100 Fibonacci sphere points in 3D around (-2, 5, 1) with radius 4.5
    std::vector<std::array<double, 3>> sphere100;
    const double golden_angle = std::numbers::pi * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < 100; ++i) {
        const double y = 1.0 - (2.0 * (i + 0.5)) / 100.0;
        const double r_xz = std::sqrt(std::max(0.0, 1.0 - y * y));
        const double theta = golden_angle * i;
        sphere100.push_back({-2.0 + 4.5 * r_xz * std::cos(theta), 5.0 + 4.5 * y,
                             1.0 + 4.5 * r_xz * std::sin(theta)});
    }
    const auto res_sp100 =
        smallest_enclosing_ball(std::span<const std::array<double, 3>>{sphere100});
    EXPECT_NEAR(res_sp100.center[0], -2.0, 1e-9);
    EXPECT_NEAR(res_sp100.center[1], 5.0, 1e-9);
    EXPECT_NEAR(res_sp100.center[2], 1.0, 1e-9);
    EXPECT_NEAR(res_sp100.radius, 4.5, 1e-9);
    for (const auto& p : sphere100) {
        EXPECT_TRUE(res_sp100.contains(p, 1e-9));
    }
}

TEST(SmallestEnclosingBallSlice3, ZeroHeapAllocationAdversarialOrderAndScaleInvariance) {
    static_assert(noexcept(smallest_enclosing_ball(std::span<const std::array<double, 2>>{})));
    static_assert(noexcept(smallest_enclosing_ball(std::span<const std::array<double, 3>>{})));
    static_assert(noexcept(smallest_enclosing_circle(std::span<const std::array<double, 2>>{})));

    // 1. Expanding spiral / sorted adversarial ordering (N = 10,000 in 2D and 3D)
    std::vector<std::array<double, 2>> spiral2;
    spiral2.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
        const double t = static_cast<double>(i) / 10000.0;
        const double r = 10.0 * t;
        const double theta = 0.1 * i;
        spiral2.push_back({r * std::cos(theta), r * std::sin(theta)});
    }
    const auto res_sp2 = smallest_enclosing_circle(spiral2);
    for (const auto& p : spiral2) {
        ASSERT_TRUE(res_sp2.contains(p, 1e-9));
    }
    EXPECT_LE(res_sp2.radius, 10.0 + 1e-9);

    std::vector<std::array<double, 3>> spiral3;
    spiral3.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
        const double t = static_cast<double>(i + 1) / 10000.0;
        const double r = 5.0 * t;
        const double phi = 0.17 * i;
        const double z = r * std::sin(0.03 * i);
        const double rho = std::sqrt(std::max(0.0, r * r - z * z));
        spiral3.push_back({rho * std::cos(phi), rho * std::sin(phi), z});
    }

#if defined(ALGOAT_TEST_CHECK_MALLINFO2)
    const auto before = mallinfo2();
#endif
    const auto res_sp3 = smallest_enclosing_ball(spiral3);
#if defined(ALGOAT_TEST_CHECK_MALLINFO2)
    const auto after = mallinfo2();
    EXPECT_EQ(after.uordblks, before.uordblks);
#endif

    for (const auto& p : spiral3) {
        ASSERT_TRUE(res_sp3.contains(p, 1e-9));
    }
    EXPECT_LE(res_sp3.radius, 5.0 + 1e-9);

    // 2. Far-from-origin translation invariance (offset = 1e8, unit radius)
    std::vector<std::array<double, 3>> far3;
    constexpr double kOffset = 1.0e8;
    for (int i = 0; i < 200; ++i) {
        const double theta = (2.0 * std::numbers::pi * i) / 200.0;
        far3.push_back({kOffset + std::cos(theta), kOffset + std::sin(theta),
                        kOffset + ((i % 2 == 0) ? 0.5 : -0.5)});
    }
    const auto res_far = smallest_enclosing_ball(std::span<const std::array<double, 3>>{far3});
    EXPECT_NEAR(res_far.center[0], kOffset, 1e-6);
    EXPECT_NEAR(res_far.center[1], kOffset, 1e-6);
    EXPECT_NEAR(res_far.center[2], kOffset, 1e-6);
    EXPECT_NEAR(res_far.radius, std::sqrt(1.25), 1e-6);
    for (const auto& p : far3) {
        EXPECT_TRUE(res_far.contains(p, 1e-8));
    }

    // 3. Tiny coordinate scale (1e-13) — must not collapse to radius 0
    std::vector<std::array<double, 2>> tiny2 = {{0.0, 0.0}, {1.0e-13, 0.0}, {0.5e-13, 0.8e-13}};
    const auto res_tiny = smallest_enclosing_circle(tiny2);
    EXPECT_GT(res_tiny.radius, 4.9e-14);
    for (const auto& p : tiny2) {
        EXPECT_TRUE(res_tiny.contains(p, 1e-9));
    }

    // 4. Huge coordinates (~1e300) — must not overflow to Inf/NaN
    std::vector<std::array<double, 3>> huge3 = {
        {-1.0e300, 0.0, 0.0}, {1.0e300, 0.0, 0.0}, {0.0, 1.0e300, 0.0}, {0.0, 0.0, -1.0e300}};
    const auto res_huge = smallest_enclosing_ball(huge3);
    EXPECT_TRUE(std::isfinite(res_huge.radius));
    EXPECT_NEAR(res_huge.radius, 1.0e300, 1.0e286);
    for (const auto& p : huge3) {
        EXPECT_TRUE(res_huge.contains(p, 1e-9));
    }

    // 5. Overloads (mutable span, std::array, float) and non-finite filtering
    std::vector<std::array<float, 3>> vec_f = {
        {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    const auto b_const = smallest_enclosing_ball(std::span<const std::array<float, 3>>{vec_f});
    const auto b_mut = smallest_enclosing_ball(std::span<std::array<float, 3>>{vec_f});
    const auto b_range = smallest_enclosing_ball(vec_f);
    EXPECT_FLOAT_EQ(b_const.radius, b_mut.radius);
    EXPECT_FLOAT_EQ(b_const.radius, b_range.radius);
    EXPECT_NEAR(b_const.radius, 1.0f, 1e-6f);
    for (const auto& p : vec_f) {
        EXPECT_TRUE(b_const.contains(p, 1e-6f));
    }

    std::array<std::array<float, 2>, 3> arr2_f{{{0.0f, 0.0f}, {3.0f, 0.0f}, {0.0f, 4.0f}}};
    const auto c_mut = smallest_enclosing_circle(std::span<std::array<float, 2>>{arr2_f});
    const auto c_range = smallest_enclosing_circle(arr2_f);
    EXPECT_FLOAT_EQ(c_mut.radius, 2.5f);
    EXPECT_FLOAT_EQ(c_range.radius, 2.5f);

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::array<double, 2>> nonfinite2 = {
        {qnan, 1.0}, {inf, -inf}, {0.0, 0.0}, {6.0, 0.0}, {3.0, qnan}};
    const auto res_nf = smallest_enclosing_circle(nonfinite2);
    EXPECT_NEAR(res_nf.center[0], 3.0, 1e-12);
    EXPECT_NEAR(res_nf.center[1], 0.0, 1e-12);
    EXPECT_NEAR(res_nf.radius, 3.0, 1e-12);

    // 6. Strided-sample misses on n > 128: unsampled [1, 2) point, subnormals, 1e78, and float
    // 1e-9f
    std::vector<std::array<double, 2>> sparse_unit(256, {0.0, 0.0});
    sparse_unit[13] = {1.5, 0.0};
    const auto res_sparse = smallest_enclosing_circle(sparse_unit);
    EXPECT_NEAR(res_sparse.center[0], 0.75, 1e-12);
    EXPECT_NEAR(res_sparse.radius, 0.75, 1e-12);

    std::vector<std::array<double, 2>> subnormal_pts(200, {0.0, 0.0});
    subnormal_pts[7] = {1.0e-315, 0.0};
    subnormal_pts[17] = {-1.0e-315, 0.0};
    const auto res_sub = smallest_enclosing_circle(subnormal_pts);
    EXPECT_TRUE(std::isfinite(res_sub.radius));
    EXPECT_NEAR(res_sub.radius, 1.0e-315, 1.0e-322);

    std::vector<std::array<double, 3>> mid_huge(200, {0.0, 0.0, 0.0});
    mid_huge[0] = {1.0e78, 0.0, 0.0};
    mid_huge[1] = {0.0, 1.0e78, 0.0};
    mid_huge[2] = {0.0, 0.0, 1.0e78};
    mid_huge[3] = {1.0e78, 1.0e78, 1.0e78};
    const auto res_mh = smallest_enclosing_ball(mid_huge);
    EXPECT_TRUE(std::isfinite(res_mh.radius));
    for (const auto& p : mid_huge) {
        EXPECT_TRUE(res_mh.contains(p, 1e-9 * std::max(1.0, res_mh.radius)));
    }

    std::vector<std::array<float, 3>> large_float(512);
    std::mt19937_64 rng_f(777ULL);
    std::uniform_real_distribution<float> dist_f(-100.0f, 100.0f);
    for (auto& p : large_float) {
        p = {dist_f(rng_f), dist_f(rng_f), dist_f(rng_f)};
    }
    const auto res_lf = smallest_enclosing_ball(std::span<const std::array<float, 3>>{large_float});
    for (const auto& p : large_float) {
        EXPECT_TRUE(res_lf.contains(p));
    }

    std::vector<std::array<double, 2>> two_large = {
        {-1.071543277662082e+37, -1.701515274893671e+38},
        {2.3509885615147286e-38, 1.2051166793193427e-43}};
    const auto res_tl = smallest_enclosing_circle(two_large);
    for (const auto& p : two_large) {
        EXPECT_TRUE(res_tl.contains(p, 0.0));
        EXPECT_TRUE(res_tl.contains(p, 1e-9));
    }
}

namespace {

[[nodiscard]] double oracle_dist2(const std::array<double, 2>& a,
                                  const std::array<double, 2>& b) noexcept {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

[[nodiscard]] double oracle_dist3(const std::array<double, 3>& a,
                                  const std::array<double, 3>& b) noexcept {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

template <std::size_t Dim, typename DistFn>
[[nodiscard]] bool oracle_covers_all(const std::array<double, Dim>& c, double r2,
                                     std::span<const std::array<double, Dim>> pts,
                                     DistFn dist_fn) noexcept {
    const double tol = r2 * (1.0 + 1e-11) + 1e-22;
    for (const auto& p : pts) {
        if (dist_fn(p, c) > tol) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] BoundingSphere<double, 2>
brute_force_seb_2d(std::span<const std::array<double, 2>> pts) noexcept {
    const std::size_t n = pts.size();
    BoundingSphere<double, 2> best{{}, std::numeric_limits<double>::infinity()};
    double best_r2 = std::numeric_limits<double>::infinity();

    auto try_sphere = [&](const std::array<double, 2>& c, double r2) noexcept {
        if (r2 >= best_r2 || !std::isfinite(r2)) {
            return;
        }
        if (oracle_covers_all<2>(c, r2, pts, oracle_dist2)) {
            best_r2 = r2;
            best.center = c;
            best.radius = std::sqrt(std::max(0.0, r2));
        }
    };

    for (std::size_t i = 0; i < n; ++i) {
        try_sphere(pts[i], 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            const std::array<double, 2> c{0.5 * (pts[i][0] + pts[j][0]),
                                          0.5 * (pts[i][1] + pts[j][1])};
            try_sphere(c, oracle_dist2(pts[i], c));
            for (std::size_t k = j + 1; k < n; ++k) {
                const double ux = pts[j][0] - pts[i][0];
                const double uy = pts[j][1] - pts[i][1];
                const double vx = pts[k][0] - pts[i][0];
                const double vy = pts[k][1] - pts[i][1];
                const double det = 2.0 * (ux * vy - uy * vx);
                if (std::abs(det) < 1e-14) {
                    continue;
                }
                const double u2 = ux * ux + uy * uy;
                const double v2 = vx * vx + vy * vy;
                const std::array<double, 2> c3{pts[i][0] + (u2 * vy - v2 * uy) / det,
                                               pts[i][1] + (ux * v2 - vx * u2) / det};
                try_sphere(c3, oracle_dist2(pts[i], c3));
            }
        }
    }
    return best;
}

[[nodiscard]] BoundingSphere<double, 3>
brute_force_seb_3d(std::span<const std::array<double, 3>> pts) noexcept {
    const std::size_t n = pts.size();
    BoundingSphere<double, 3> best{{}, std::numeric_limits<double>::infinity()};
    double best_r2 = std::numeric_limits<double>::infinity();

    auto try_sphere = [&](const std::array<double, 3>& c, double r2) noexcept {
        if (r2 >= best_r2 || !std::isfinite(r2)) {
            return;
        }
        if (oracle_covers_all<3>(c, r2, pts, oracle_dist3)) {
            best_r2 = r2;
            best.center = c;
            best.radius = std::sqrt(std::max(0.0, r2));
        }
    };

    for (std::size_t i = 0; i < n; ++i) {
        try_sphere(pts[i], 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            const std::array<double, 3> c2{0.5 * (pts[i][0] + pts[j][0]),
                                           0.5 * (pts[i][1] + pts[j][1]),
                                           0.5 * (pts[i][2] + pts[j][2])};
            try_sphere(c2, oracle_dist3(pts[i], c2));
            for (std::size_t k = j + 1; k < n; ++k) {
                const std::array<double, 3> u{pts[j][0] - pts[i][0], pts[j][1] - pts[i][1],
                                              pts[j][2] - pts[i][2]};
                const std::array<double, 3> v{pts[k][0] - pts[i][0], pts[k][1] - pts[i][1],
                                              pts[k][2] - pts[i][2]};
                const double cx = u[1] * v[2] - u[2] * v[1];
                const double cy = u[2] * v[0] - u[0] * v[2];
                const double cz = u[0] * v[1] - u[1] * v[0];
                const double denom = 2.0 * (cx * cx + cy * cy + cz * cz);
                if (denom > 1e-18) {
                    const double u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                    const double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    const double uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
                    const double a = v2 * (u2 - uv) / denom;
                    const double b = u2 * (v2 - uv) / denom;
                    const std::array<double, 3> c3{pts[i][0] + a * u[0] + b * v[0],
                                                   pts[i][1] + a * u[1] + b * v[1],
                                                   pts[i][2] + a * u[2] + b * v[2]};
                    try_sphere(c3, oracle_dist3(pts[i], c3));
                }
                for (std::size_t l = k + 1; l < n; ++l) {
                    const std::array<double, 3> w{pts[l][0] - pts[i][0], pts[l][1] - pts[i][1],
                                                  pts[l][2] - pts[i][2]};
                    const std::array<double, 3> vw{v[1] * w[2] - v[2] * w[1],
                                                   v[2] * w[0] - v[0] * w[2],
                                                   v[0] * w[1] - v[1] * w[0]};
                    const std::array<double, 3> wu{w[1] * u[2] - w[2] * u[1],
                                                   w[2] * u[0] - w[0] * u[2],
                                                   w[0] * u[1] - w[1] * u[0]};
                    const std::array<double, 3> uv_c{cx, cy, cz};
                    const double det = 2.0 * (u[0] * vw[0] + u[1] * vw[1] + u[2] * vw[2]);
                    if (std::abs(det) <= 1e-14) {
                        continue;
                    }
                    const double u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                    const double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    const double w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                    const std::array<double, 3> c4{
                        pts[i][0] + (u2 * vw[0] + v2 * wu[0] + w2 * uv_c[0]) / det,
                        pts[i][1] + (u2 * vw[1] + v2 * wu[1] + w2 * uv_c[1]) / det,
                        pts[i][2] + (u2 * vw[2] + v2 * wu[2] + w2 * uv_c[2]) / det};
                    try_sphere(c4, oracle_dist3(pts[i], c4));
                }
            }
        }
    }
    return best;
}

[[nodiscard]] bool
kkt_center_in_boundary_hull_2d(const BoundingSphere<double, 2>& s,
                               std::span<const std::array<double, 2>> pts) noexcept {
    if (s.radius == 0.0) {
        return true;
    }
    const double tol = 1e-6 * std::max(1.0, s.radius);
    std::vector<double> angles;
    for (const auto& p : pts) {
        const double dist = std::sqrt(oracle_dist2(p, s.center));
        if (std::abs(dist - s.radius) <= tol) {
            angles.push_back(std::atan2(p[1] - s.center[1], p[0] - s.center[0]));
        }
    }
    if (angles.size() < 2) {
        return false;
    }
    std::sort(angles.begin(), angles.end());
    double max_gap = (2.0 * std::numbers::pi + angles.front()) - angles.back();
    for (std::size_t i = 1; i < angles.size(); ++i) {
        max_gap = std::max(max_gap, angles[i] - angles[i - 1]);
    }
    return max_gap <= std::numbers::pi + 1e-4;
}

[[nodiscard]] bool
kkt_center_in_boundary_hull_3d(const BoundingSphere<double, 3>& s,
                               std::span<const std::array<double, 3>> pts) noexcept {
    if (s.radius == 0.0) {
        return true;
    }
    const double tol = 1e-5 * std::max(1.0, s.radius);
    std::vector<std::array<double, 3>> b;
    for (const auto& p : pts) {
        const double dist = std::sqrt(oracle_dist3(p, s.center));
        if (std::abs(dist - s.radius) <= tol) {
            b.push_back(p);
        }
    }
    if (b.size() < 2) {
        return false;
    }
    const std::size_t m = b.size();
    const double eps = 1e-4 * std::max(1.0, s.radius);
    // Check 2-point diameter chord containing center
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            const std::array<double, 3> mid{0.5 * (b[i][0] + b[j][0]), 0.5 * (b[i][1] + b[j][1]),
                                            0.5 * (b[i][2] + b[j][2])};
            if (std::sqrt(oracle_dist3(mid, s.center)) <= eps) {
                return true;
            }
        }
    }
    // Check 3-point triangle containing center in its affine plane
    auto sub = [](const std::array<double, 3>& a, const std::array<double, 3>& q) {
        return std::array<double, 3>{a[0] - q[0], a[1] - q[1], a[2] - q[2]};
    };
    auto dot3 = [](const std::array<double, 3>& a, const std::array<double, 3>& q) {
        return a[0] * q[0] + a[1] * q[1] + a[2] * q[2];
    };
    auto cross3 = [](const std::array<double, 3>& a, const std::array<double, 3>& q) {
        return std::array<double, 3>{a[1] * q[2] - a[2] * q[1], a[2] * q[0] - a[0] * q[2],
                                     a[0] * q[1] - a[1] * q[0]};
    };

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            for (std::size_t k = j + 1; k < m; ++k) {
                const auto u = sub(b[j], b[i]);
                const auto v = sub(b[k], b[i]);
                const auto w = sub(s.center, b[i]);
                const auto n = cross3(u, v);
                const double n2 = dot3(n, n);
                if (n2 <= 1e-14) {
                    continue;
                }
                const double plane_dist = std::abs(dot3(w, n)) / std::sqrt(n2);
                if (plane_dist <= eps) {
                    const double b_k = dot3(cross3(u, w), n) / n2;
                    const double b_j = dot3(cross3(w, v), n) / n2;
                    const double b_i = 1.0 - b_j - b_k;
                    if (b_i >= -1e-3 && b_j >= -1e-3 && b_k >= -1e-3) {
                        return true;
                    }
                }
            }
        }
    }
    // Check 4-point tetrahedron containing center via signed volume ratios
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            for (std::size_t k = j + 1; k < m; ++k) {
                for (std::size_t l = k + 1; l < m; ++l) {
                    auto signed_tet_volume =
                        [&](const std::array<double, 3>& a, const std::array<double, 3>& p,
                            const std::array<double, 3>& q, const std::array<double, 3>& d) {
                            return dot3(sub(a, d), cross3(sub(p, d), sub(q, d)));
                        };
                    const double v_all = signed_tet_volume(b[i], b[j], b[k], b[l]);
                    if (std::abs(v_all) <= 1e-14) {
                        continue;
                    }
                    const double w0 = signed_tet_volume(s.center, b[j], b[k], b[l]) / v_all;
                    const double w1 = signed_tet_volume(b[i], s.center, b[k], b[l]) / v_all;
                    const double w2 = signed_tet_volume(b[i], b[j], s.center, b[l]) / v_all;
                    const double w3 = signed_tet_volume(b[i], b[j], b[k], s.center) / v_all;
                    if (w0 >= -1e-3 && w1 >= -1e-3 && w2 >= -1e-3 && w3 >= -1e-3) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

} // namespace

TEST(SmallestEnclosingBallSlice4, Randomized1000InstancesOracleAndKKTOptimality2DAnd3D) {
    std::mt19937_64 rng(20261117ULL);
    std::uniform_real_distribution<double> coord_dist(-100.0, 100.0);
    std::uniform_int_distribution<int> grid_dist(-4, 4);
    std::uniform_int_distribution<std::size_t> size_small(2, 24);
    std::uniform_int_distribution<std::size_t> size_large(129, 2000);

    // 500 2D trials (brute force on small N + KKT convex hull optimality on all N)
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = (trial < 400) ? size_small(rng) : size_large(rng);
        std::vector<std::array<double, 2>> pts(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (trial % 3 == 0) {
                pts[i] = {static_cast<double>(grid_dist(rng)), static_cast<double>(grid_dist(rng))};
            } else {
                pts[i] = {coord_dist(rng), coord_dist(rng)};
            }
        }

        const auto got = smallest_enclosing_circle(std::span<const std::array<double, 2>>{pts});
        for (const auto& p : pts) {
            ASSERT_TRUE(got.contains(p, 1e-9)) << "trial=" << trial;
        }
        ASSERT_TRUE(kkt_center_in_boundary_hull_2d(got, pts)) << "2D KKT failed trial=" << trial;

        if (n <= 24) {
            const auto expected = brute_force_seb_2d(pts);
            ASSERT_NEAR(got.radius, expected.radius, 1e-9) << "trial=" << trial;
            ASSERT_NEAR(got.center[0], expected.center[0], 1e-8) << "trial=" << trial;
            ASSERT_NEAR(got.center[1], expected.center[1], 1e-8) << "trial=" << trial;
        }
    }

    // 500 3D trials (brute force on small N + KKT convex hull optimality on all N)
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n =
            (trial < 400) ? std::min<std::size_t>(size_small(rng), 18) : size_large(rng);
        std::vector<std::array<double, 3>> pts(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (trial % 3 == 0) {
                pts[i] = {static_cast<double>(grid_dist(rng)), static_cast<double>(grid_dist(rng)),
                          static_cast<double>(grid_dist(rng))};
            } else {
                pts[i] = {coord_dist(rng), coord_dist(rng), coord_dist(rng)};
            }
        }

        const auto got = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
        for (const auto& p : pts) {
            ASSERT_TRUE(got.contains(p, 1e-9)) << "trial=" << trial;
        }
        ASSERT_TRUE(kkt_center_in_boundary_hull_3d(got, pts)) << "3D KKT failed trial=" << trial;

        if (n <= 18) {
            const auto expected = brute_force_seb_3d(pts);
            ASSERT_NEAR(got.radius, expected.radius, 1e-9) << "trial=" << trial;
            ASSERT_NEAR(got.center[0], expected.center[0], 1e-7) << "trial=" << trial;
            ASSERT_NEAR(got.center[1], expected.center[1], 1e-7) << "trial=" << trial;
            ASSERT_NEAR(got.center[2], expected.center[2], 1e-7) << "trial=" << trial;
        }
    }
}

} // namespace algoat::geometry::testing
