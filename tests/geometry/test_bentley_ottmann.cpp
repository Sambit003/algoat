/**
 * @file test_bentley_ottmann.cpp
 * @brief Unit tests for Bentley-Ottmann 2D line segment intersection sweep-line.
 */

#include "algoat/geometry/bentley_ottmann.hpp"
#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace algoat::geometry::testing {

namespace {

[[nodiscard]] bool seg_contains_collinear(const Point2D& a, const Point2D& b,
                                          const Point2D& q) noexcept {
    return q.x >= std::min(a.x, b.x) && q.x <= std::max(a.x, b.x) && q.y >= std::min(a.y, b.y) &&
           q.y <= std::max(a.y, b.y);
}

[[nodiscard]] bool pt_less(const Point2D& a, const Point2D& b) noexcept {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
}

[[nodiscard]] bool independent_segments_intersect(const Point2D& p0, const Point2D& p1,
                                                  const Point2D& q0, const Point2D& q1,
                                                  Point2D& out_ip) noexcept {
    if (std::max(p0.x, p1.x) < std::min(q0.x, q1.x) ||
        std::max(q0.x, q1.x) < std::min(p0.x, p1.x) ||
        std::max(p0.y, p1.y) < std::min(q0.y, q1.y) ||
        std::max(q0.y, q1.y) < std::min(p0.y, p1.y)) {
        return false;
    }

    const bool p_pt = (p0.x == p1.x && p0.y == p1.y);
    const bool q_pt = (q0.x == q1.x && q0.y == q1.y);
    if (p_pt && q_pt) {
        if (p0.x == q0.x && p0.y == q0.y) {
            out_ip = p0;
            return true;
        }
        return false;
    }
    if (p_pt) {
        if (orient2d(q0, q1, p0) == Orientation::Collinear && seg_contains_collinear(q0, q1, p0)) {
            out_ip = p0;
            return true;
        }
        return false;
    }
    if (q_pt) {
        if (orient2d(p0, p1, q0) == Orientation::Collinear && seg_contains_collinear(p0, p1, q0)) {
            out_ip = q0;
            return true;
        }
        return false;
    }

    const Orientation o1 = orient2d(p0, p1, q0);
    const Orientation o2 = orient2d(p0, p1, q1);
    const Orientation o3 = orient2d(q0, q1, p0);
    const Orientation o4 = orient2d(q0, q1, p1);

    if (o1 == Orientation::Collinear && o2 == Orientation::Collinear &&
        o3 == Orientation::Collinear && o4 == Orientation::Collinear) {
        const Point2D a_lo = pt_less(p0, p1) ? p0 : p1;
        const Point2D a_hi = pt_less(p0, p1) ? p1 : p0;
        const Point2D b_lo = pt_less(q0, q1) ? q0 : q1;
        const Point2D b_hi = pt_less(q0, q1) ? q1 : q0;
        const Point2D lo = pt_less(a_lo, b_lo) ? b_lo : a_lo;
        const Point2D hi = pt_less(a_hi, b_hi) ? a_hi : b_hi;
        if (pt_less(hi, lo)) {
            return false;
        }
        out_ip = lo;
        return true;
    }

    if (o1 == Orientation::Collinear && seg_contains_collinear(p0, p1, q0)) {
        out_ip = q0;
        return true;
    }
    if (o2 == Orientation::Collinear && seg_contains_collinear(p0, p1, q1)) {
        out_ip = q1;
        return true;
    }
    if (o3 == Orientation::Collinear && seg_contains_collinear(q0, q1, p0)) {
        out_ip = p0;
        return true;
    }
    if (o4 == Orientation::Collinear && seg_contains_collinear(q0, q1, p1)) {
        out_ip = p1;
        return true;
    }

    if (o1 == o2 || o3 == o4) {
        return false;
    }

    // Proper crossing using standard 2D determinant in scaled coordinates
    const double dx1 = p1.x - p0.x;
    const double dy1 = p1.y - p0.y;
    const double dx2 = q1.x - q0.x;
    const double dy2 = q1.y - q0.y;
    const double denom = dx1 * dy2 - dy1 * dx2;
    if (denom != 0.0 && std::isfinite(denom)) {
        const double t = ((q0.x - p0.x) * dy2 - (q0.y - p0.y) * dx2) / denom;
        if (std::isfinite(t)) {
            out_ip = {p0.x + std::clamp(t, 0.0, 1.0) * dx1, p0.y + std::clamp(t, 0.0, 1.0) * dy1};
            return true;
        }
    }
    out_ip = {0.5 * (std::max(std::min(p0.x, p1.x), std::min(q0.x, q1.x)) +
                     std::min(std::max(p0.x, p1.x), std::max(q0.x, q1.x))),
              0.5 * (std::max(std::min(p0.y, p1.y), std::min(q0.y, q1.y)) +
                     std::min(std::max(p0.y, p1.y), std::max(q0.y, q1.y)))};
    return true;
}

template <typename T>
std::vector<IntersectionResult<T>>
brute_force_intersections(std::span<const LineSegment<T>> segments) {
    std::vector<IntersectionResult<T>> out;
    const std::size_t n = segments.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D p0{static_cast<double>(segments[i].start[0]),
                         static_cast<double>(segments[i].start[1])};
        const Point2D p1{static_cast<double>(segments[i].end[0]),
                         static_cast<double>(segments[i].end[1])};
        if (!std::isfinite(p0.x) || !std::isfinite(p0.y) || !std::isfinite(p1.x) ||
            !std::isfinite(p1.y)) {
            continue;
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            const Point2D q0{static_cast<double>(segments[j].start[0]),
                             static_cast<double>(segments[j].start[1])};
            const Point2D q1{static_cast<double>(segments[j].end[0]),
                             static_cast<double>(segments[j].end[1])};
            if (!std::isfinite(q0.x) || !std::isfinite(q0.y) || !std::isfinite(q1.x) ||
                !std::isfinite(q1.y)) {
                continue;
            }
            Point2D ip{};
            if (independent_segments_intersect(p0, p1, q0, q1, ip)) {
                out.push_back(
                    IntersectionResult<T>{{static_cast<T>(ip.x), static_cast<T>(ip.y)}, i, j});
            }
        }
    }
    return out;
}

template <typename T>
void expect_matches_brute_force(std::span<const LineSegment<T>> segs, const char* context = "") {
    const auto got = bentley_ottmann(segs);
    const auto expected = brute_force_intersections(segs);
    ASSERT_EQ(got.size(), expected.size()) << context;
    ASSERT_EQ(count_intersections(segs), expected.size()) << context;
    for (std::size_t k = 0; k < expected.size(); ++k) {
        EXPECT_EQ(got[k].segment1_index, expected[k].segment1_index) << context << " k=" << k;
        EXPECT_EQ(got[k].segment2_index, expected[k].segment2_index) << context << " k=" << k;
        const auto& s1 = segs[got[k].segment1_index];
        const auto& s2 = segs[got[k].segment2_index];
        const double max_coord = std::max<double>(
            {1.0, std::abs(static_cast<double>(s1.start[0])),
             std::abs(static_cast<double>(s1.start[1])), std::abs(static_cast<double>(s1.end[0])),
             std::abs(static_cast<double>(s1.end[1])), std::abs(static_cast<double>(s2.start[0])),
             std::abs(static_cast<double>(s2.start[1])), std::abs(static_cast<double>(s2.end[0])),
             std::abs(static_cast<double>(s2.end[1]))});
        const double tol = std::is_same_v<T, float> ? (1e-4 * max_coord) : (1e-10 * max_coord);
        const double gx = static_cast<double>(got[k].point[0]);
        const double gy = static_cast<double>(got[k].point[1]);
        const double ex = static_cast<double>(expected[k].point[0]);
        const double ey = static_cast<double>(expected[k].point[1]);
        EXPECT_TRUE(std::isfinite(gx) && std::isfinite(gy)) << context << " k=" << k;
        if (max_coord < 1e30) {
            EXPECT_NEAR(gx, ex, tol) << context << " k=" << k;
            EXPECT_NEAR(gy, ey, tol) << context << " k=" << k;
        }
    }
}

} // namespace

TEST(BentleyOttmannSlice1, EmptySingleDisjointAndBasicCross) {
    std::vector<LineSegment<double>> empty;
    EXPECT_TRUE(bentley_ottmann(std::span<const LineSegment<double>>{empty}).empty());
    EXPECT_EQ(count_intersections(std::span<const LineSegment<double>>{empty}), 0u);

    std::vector<LineSegment<double>> single = {{{0.0, 0.0}, {10.0, 10.0}}};
    EXPECT_TRUE(bentley_ottmann(std::span<const LineSegment<double>>{single}).empty());
    EXPECT_EQ(count_intersections(std::span<const LineSegment<double>>{single}), 0u);

    std::vector<LineSegment<double>> disjoint = {
        {{0.0, 0.0}, {4.0, 4.0}},
        {{6.0, 0.0}, {10.0, 4.0}},
    };
    EXPECT_TRUE(bentley_ottmann(std::span<const LineSegment<double>>{disjoint}).empty());
    EXPECT_EQ(count_intersections(std::span<const LineSegment<double>>{disjoint}), 0u);

    std::vector<LineSegment<double>> cross = {
        {{0.0, 0.0}, {10.0, 10.0}},
        {{0.0, 10.0}, {10.0, 0.0}},
    };
    const auto res = bentley_ottmann(std::span<const LineSegment<double>>{cross});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].segment1_index, 0u);
    EXPECT_EQ(res[0].segment2_index, 1u);
    EXPECT_DOUBLE_EQ(res[0].point[0], 5.0);
    EXPECT_DOUBLE_EQ(res[0].point[1], 5.0);
    EXPECT_EQ(count_intersections(std::span<const LineSegment<double>>{cross}), 1u);
}

TEST(BentleyOttmannSlice2, VerticalSegmentsZeroLengthSharedEndpointsAndTJunctions) {
    std::vector<LineSegment<double>> segs = {
        {{5.0, 0.0}, {5.0, 10.0}},  // 0: vertical x=5, y in [0, 10]
        {{0.0, 5.0}, {10.0, 5.0}},  // 1: horizontal y=5
        {{0.0, 0.0}, {10.0, 10.0}}, // 2: diagonal crossing (5, 5)
        {{5.0, 4.0}, {5.0, 12.0}},  // 3: vertical overlapping segment 0 at x=5, y in [4, 10]
        {{5.0, 2.0}, {5.0, 2.0}},   // 4: degenerate zero-length point on segment 0 at (5, 2)
        {{0.0, 0.0}, {5.0, 0.0}},   // 5: ends at bottom of segment 0 at (5, 0)
        {{5.0, 10.0}, {9.0, 10.0}}, // 6: starts at top of segment 0 at (5, 10)
        {{2.0, 2.0}, {6.0, 2.0}},   // 7: passes through point segment 4 at (5, 2)
        {{2.0, 8.0}, {8.0, 8.0}},   // 8: horizontal y=8
        {{5.0, 8.0}, {7.0, 12.0}},  // 9: T-junction starting on segment 8 & 0 & 3 at (5, 8)
    };
    expect_matches_brute_force<double>(segs, "VerticalTJunctionMix");

    std::vector<LineSegment<double>> polygon_chain = {
        {{0.0, 0.0}, {4.0, 0.0}},
        {{4.0, 0.0}, {4.0, 4.0}},
        {{4.0, 4.0}, {0.0, 4.0}},
        {{0.0, 4.0}, {0.0, 0.0}},
    };
    expect_matches_brute_force<double>(polygon_chain, "SquareBoundaryChain");
    EXPECT_EQ(count_intersections(polygon_chain), 4u);
}

TEST(BentleyOttmannSlice3, CollinearOverlapsMultiSegmentPencilsAndOverloads) {
    std::vector<LineSegment<double>> collinear_bundle = {
        {{0.0, 0.0}, {10.0, 10.0}},   {{2.0, 2.0}, {8.0, 8.0}},   {{4.0, 4.0}, {12.0, 12.0}},
        {{10.0, 10.0}, {14.0, 14.0}}, {{0.0, 10.0}, {10.0, 0.0}}, {{0.0, 5.0}, {10.0, 5.0}},
        {{5.0, -2.0}, {5.0, 12.0}},   {{5.0, 5.0}, {15.0, 5.0}},  {{-2.0, 5.0}, {5.0, 5.0}},
    };
    expect_matches_brute_force<double>(collinear_bundle, "CollinearBundleAndCrossing");

    std::vector<LineSegment<double>> pencil = {
        {{0.0, 0.0}, {10.0, 10.0}}, {{0.0, 2.0}, {10.0, 8.0}},  {{0.0, 5.0}, {10.0, 5.0}},
        {{0.0, 8.0}, {10.0, 2.0}},  {{0.0, 10.0}, {10.0, 0.0}}, {{5.0, 0.0}, {5.0, 10.0}},
        {{2.0, 1.0}, {5.0, 5.0}},   {{5.0, 5.0}, {9.0, 11.0}},
    };
    expect_matches_brute_force<double>(pencil, "EightWayPencilAt55");
    EXPECT_EQ(count_intersections(pencil), 28u);

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<LineSegment<double>> nonfinite = {
        {{0.0, 0.0}, {10.0, 10.0}},
        {{0.0, 10.0}, {10.0, 0.0}},
        {{qnan, 0.0}, {5.0, 5.0}},
        {{0.0, inf}, {5.0, 5.0}},
    };
    expect_matches_brute_force<double>(nonfinite, "NonFiniteCoordinates");

    std::vector<LineSegment<float>> segs_f = {
        {{0.0f, 0.0f}, {6.0f, 6.0f}},
        {{0.0f, 6.0f}, {6.0f, 0.0f}},
        {{3.0f, -1.0f}, {3.0f, 7.0f}},
    };
    auto res_vec = bentley_ottmann(segs_f);
    auto res_mut = bentley_ottmann(std::span<LineSegment<float>>{segs_f});
    EXPECT_EQ(res_vec, res_mut);
    EXPECT_EQ(count_intersections(segs_f), 3u);
    expect_matches_brute_force<float>(segs_f, "FloatSegments");

    std::vector<LineSegment<double>> extreme_scale = {
        {{3.587324068671532e-43, 0.0}, {-1.7077498847817374e+38, 2.0282831177758272e-38}},
        {{0.0, -2.3867795255820682e+30}, {-3.880299964402159e-34, 3.489747452705202e-39}},
        {{0.0, 4.519934980173775e+35}, {-2.2630148322885632e+17, -3.8519609078202455e-34}},
        {{-8.216236232862422e-34, 5.428372276841178e-25},
         {1960905436823552.0, -8.217214280896505e-34}},
    };
    expect_matches_brute_force<double>(extreme_scale, "ExtremeDynamicRangeRegression");

    // Near-limit +/-1e308 coordinates verifying overflow-free delta/eval_y handling
    std::vector<LineSegment<double>> huge_coords = {
        {{-1e308, -1e308}, {1e308, 1e308}},
        {{-1e308, 1e308}, {1e308, -1e308}},
        {{0.0, -1e308}, {0.0, 1e308}},
    };
    const auto huge_res = bentley_ottmann(std::span<const LineSegment<double>>{huge_coords});
    ASSERT_EQ(huge_res.size(), 3u);
    for (const auto& r : huge_res) {
        EXPECT_TRUE(std::isfinite(r.point[0]) && std::isfinite(r.point[1]));
    }
}

TEST(BentleyOttmannSlice4, ExactEquivalence10000RandomSegmentSets) {
    std::mt19937_64 rng(20261116ULL);
    constexpr int kTotalTrials = 10000;

    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    std::uniform_int_distribution<int> grid_dist(-6, 6);
    std::uniform_int_distribution<int> small_n(2, 64);
    std::uniform_int_distribution<int> med_n(65, 150);
    std::uniform_int_distribution<int> large_n(151, 5000);

    for (int trial = 0; trial < kTotalTrials; ++trial) {
        std::size_t n = 0;
        if (trial == 0) {
            n = 2;
        } else if (trial == 1) {
            n = 5000;
        } else if (trial < 9000) {
            n = static_cast<std::size_t>(small_n(rng));
        } else if (trial < 9920) {
            n = static_cast<std::size_t>(med_n(rng));
        } else {
            n = static_cast<std::size_t>(large_n(rng));
        }

        const int mode = (n > 150) ? ((trial % 2 == 0) ? 0 : 3) : (trial % 5);
        const double span_scale = (n > 150) ? (1000.0 / std::sqrt(static_cast<double>(n))) : 20.0;
        std::vector<LineSegment<double>> segs;
        segs.reserve(n);

        for (std::size_t i = 0; i < n; ++i) {
            if (mode == 0) {
                const double x0 = unit_dist(rng) * 1000.0;
                const double y0 = unit_dist(rng) * 1000.0;
                const double dx = (unit_dist(rng) - 0.5) * span_scale;
                const double dy = (unit_dist(rng) - 0.5) * span_scale;
                segs.push_back({{x0, y0}, {x0 + dx, y0 + dy}});
            } else if (mode == 1) {
                segs.push_back({{unit_dist(rng) * 100.0, unit_dist(rng) * 100.0},
                                {unit_dist(rng) * 100.0, unit_dist(rng) * 100.0}});
            } else if (mode == 2) {
                segs.push_back(
                    {{static_cast<double>(grid_dist(rng)), static_cast<double>(grid_dist(rng))},
                     {static_cast<double>(grid_dist(rng)), static_cast<double>(grid_dist(rng))}});
            } else if (mode == 3) {
                const int kind = static_cast<int>(i % 3);
                const double x0 = (n > 150) ? (unit_dist(rng) * 1000.0)
                                            : (static_cast<double>(grid_dist(rng)) * 2.0);
                const double y0 =
                    (n > 150) ? (unit_dist(rng) * 1000.0) : (unit_dist(rng) * 20.0 - 10.0);
                const double len = (n > 150) ? span_scale : 8.0;
                if (kind == 0) {
                    segs.push_back({{x0, y0}, {x0, y0 + unit_dist(rng) * len}});
                } else if (kind == 1) {
                    segs.push_back({{x0, y0}, {x0 + 0.1 + unit_dist(rng) * len, y0}});
                } else {
                    segs.push_back(
                        {{x0, y0},
                         {x0 + (unit_dist(rng) - 0.5) * len, y0 + (unit_dist(rng) - 0.5) * len}});
                }
            } else {
                constexpr double kOffset = 1.0e8;
                const double x0 = kOffset + unit_dist(rng) * 50.0;
                const double y0 = kOffset + unit_dist(rng) * 50.0;
                const double dx = (unit_dist(rng) - 0.5) * 12.0;
                const double dy = (unit_dist(rng) - 0.5) * 1e-9;
                segs.push_back({{x0, y0}, {x0 + dx, y0 + dy}});
            }
        }

        const auto got = bentley_ottmann(std::span<const LineSegment<double>>{segs});
        const auto expected = brute_force_intersections<double>(segs);
        ASSERT_EQ(got.size(), expected.size())
            << "trial=" << trial << " mode=" << mode << " n=" << n;
        ASSERT_EQ(count_intersections(std::span<const LineSegment<double>>{segs}), expected.size())
            << "trial=" << trial << " mode=" << mode << " n=" << n;
        for (std::size_t k = 0; k < expected.size(); ++k) {
            ASSERT_EQ(got[k].segment1_index, expected[k].segment1_index)
                << "trial=" << trial << " k=" << k;
            ASSERT_EQ(got[k].segment2_index, expected[k].segment2_index)
                << "trial=" << trial << " k=" << k;
            ASSERT_NEAR(got[k].point[0], expected[k].point[0], 1e-7)
                << "trial=" << trial << " k=" << k;
            ASSERT_NEAR(got[k].point[1], expected[k].point[1], 1e-7)
                << "trial=" << trial << " k=" << k;
        }
    }
}

} // namespace algoat::geometry::testing
