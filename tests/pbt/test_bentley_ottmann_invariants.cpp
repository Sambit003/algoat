/**
 * @file test_bentley_ottmann_invariants.cpp
 * @brief RapidCheck Property-Based Tests for Bentley-Ottmann line segment intersection.
 */

#include "algoat/geometry/bentley_ottmann.hpp"
#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <span>
#include <utility>
#include <vector>

using namespace algoat::geometry;

namespace {

[[nodiscard]] bool pt_less(const Point2D& a, const Point2D& b) noexcept {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
}

[[nodiscard]] bool in_box(const Point2D& a, const Point2D& b, const Point2D& q) noexcept {
    return q.x >= std::min(a.x, b.x) && q.x <= std::max(a.x, b.x) && q.y >= std::min(a.y, b.y) &&
           q.y <= std::max(a.y, b.y);
}

[[nodiscard]] bool independent_intersect(const Point2D& p0, const Point2D& p1, const Point2D& q0,
                                         const Point2D& q1, Point2D& out_ip) noexcept {
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
        if (orient2d(q0, q1, p0) == Orientation::Collinear && in_box(q0, q1, p0)) {
            out_ip = p0;
            return true;
        }
        return false;
    }
    if (q_pt) {
        if (orient2d(p0, p1, q0) == Orientation::Collinear && in_box(p0, p1, q0)) {
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
        const Point2D lo = pt_less(pt_less(p0, p1) ? p0 : p1, pt_less(q0, q1) ? q0 : q1)
                               ? (pt_less(q0, q1) ? q0 : q1)
                               : (pt_less(p0, p1) ? p0 : p1);
        const Point2D hi = pt_less(pt_less(p0, p1) ? p1 : p0, pt_less(q0, q1) ? q1 : q0)
                               ? (pt_less(p0, p1) ? p1 : p0)
                               : (pt_less(q0, q1) ? q1 : q0);
        if (pt_less(hi, lo)) {
            return false;
        }
        out_ip = lo;
        return true;
    }
    if (o1 == Orientation::Collinear && in_box(p0, p1, q0)) {
        out_ip = q0;
        return true;
    }
    if (o2 == Orientation::Collinear && in_box(p0, p1, q1)) {
        out_ip = q1;
        return true;
    }
    if (o3 == Orientation::Collinear && in_box(q0, q1, p0)) {
        out_ip = p0;
        return true;
    }
    if (o4 == Orientation::Collinear && in_box(q0, q1, p1)) {
        out_ip = p1;
        return true;
    }
    if (o1 == o2 || o3 == o4) {
        return false;
    }

    const long double dx1 = static_cast<long double>(p1.x) - static_cast<long double>(p0.x);
    const long double dy1 = static_cast<long double>(p1.y) - static_cast<long double>(p0.y);
    const long double dx2 = static_cast<long double>(q1.x) - static_cast<long double>(q0.x);
    const long double dy2 = static_cast<long double>(q1.y) - static_cast<long double>(q0.y);
    const long double denom = dx1 * dy2 - dy1 * dx2;
    const long double t =
        (denom != 0.0L)
            ? std::clamp(((static_cast<long double>(q0.x) - static_cast<long double>(p0.x)) * dy2 -
                          (static_cast<long double>(q0.y) - static_cast<long double>(p0.y)) * dx2) /
                             denom,
                         0.0L, 1.0L)
            : 0.5L;
    out_ip = {static_cast<double>(static_cast<long double>(p0.x) + t * dx1),
              static_cast<double>(static_cast<long double>(p0.y) + t * dy1)};
    return true;
}

template <typename T>
std::vector<IntersectionResult<T>> brute_force_oracle(std::span<const LineSegment<T>> segments) {
    std::vector<IntersectionResult<T>> out;
    const std::size_t n = segments.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D p0{static_cast<double>(segments[i].start[0]),
                         static_cast<double>(segments[i].start[1])};
        const Point2D p1{static_cast<double>(segments[i].end[0]),
                         static_cast<double>(segments[i].end[1])};
        for (std::size_t j = i + 1; j < n; ++j) {
            const Point2D q0{static_cast<double>(segments[j].start[0]),
                             static_cast<double>(segments[j].start[1])};
            const Point2D q1{static_cast<double>(segments[j].end[0]),
                             static_cast<double>(segments[j].end[1])};
            Point2D ip{};
            if (independent_intersect(p0, p1, q0, q1, ip)) {
                out.push_back(
                    IntersectionResult<T>{{static_cast<T>(ip.x), static_cast<T>(ip.y)}, i, j});
            }
        }
    }
    return out;
}

template <typename T> void assert_matches_oracle(std::span<const LineSegment<T>> segs) {
    const auto got = bentley_ottmann(segs);
    const auto expected = brute_force_oracle<T>(segs);
    RC_ASSERT(got.size() == expected.size());
    RC_ASSERT(count_intersections(segs) == expected.size());
    const double abs_tol = std::is_same_v<T, float> ? 1e-4 : 1e-9;
    const double rel_tol = std::is_same_v<T, float> ? 1e-5 : 1e-12;
    for (std::size_t k = 0; k < expected.size(); ++k) {
        RC_ASSERT(got[k].segment1_index == expected[k].segment1_index);
        RC_ASSERT(got[k].segment2_index == expected[k].segment2_index);
        const double gx = static_cast<double>(got[k].point[0]);
        const double ex = static_cast<double>(expected[k].point[0]);
        const double gy = static_cast<double>(got[k].point[1]);
        const double ey = static_cast<double>(expected[k].point[1]);
        const double tol_x = std::max(abs_tol, rel_tol * std::max(std::abs(gx), std::abs(ex)));
        const double tol_y = std::max(abs_tol, rel_tol * std::max(std::abs(gy), std::abs(ey)));
        RC_ASSERT(std::abs(gx - ex) <= tol_x);
        RC_ASSERT(std::abs(gy - ey) <= tol_y);
    }
}

template <typename T>
std::vector<LineSegment<T>> build_segments(const std::vector<std::array<int, 4>>& raw, T scale) {
    const std::size_t n = std::min<std::size_t>(raw.size(), 120);
    std::vector<LineSegment<T>> segs;
    segs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        segs.push_back({{static_cast<T>(raw[i][0]) * scale, static_cast<T>(raw[i][1]) * scale},
                        {static_cast<T>(raw[i][2]) * scale, static_cast<T>(raw[i][3]) * scale}});
    }
    return segs;
}

} // namespace

RC_GTEST_PROP(BentleyOttmannPBT, MatchesBruteForceOnRandomSegments,
              (const std::vector<std::array<int, 4>>& raw_segs)) {
    const auto segs = build_segments<double>(raw_segs, 0.125);
    assert_matches_oracle<double>(segs);
}

RC_GTEST_PROP(BentleyOttmannPBT, MatchesBruteForceWithVerticalCollinearAndMultiPointDegeneracies,
              (const std::vector<std::array<int8_t, 4>>& raw_grid)) {
    const std::size_t n = std::min<std::size_t>(raw_grid.size(), 90);
    std::vector<LineSegment<double>> segs;
    segs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x1 = static_cast<double>(raw_grid[i][0] % 4);
        const double y1 = static_cast<double>(raw_grid[i][1] % 4);
        const double x2 = static_cast<double>(raw_grid[i][2] % 4);
        const double y2 = static_cast<double>(raw_grid[i][3] % 4);
        segs.push_back({{x1, y1}, {x2, y2}});
    }
    assert_matches_oracle<double>(segs);
}

RC_GTEST_PROP(BentleyOttmannPBT, EndpointFlipAndPermutationInvariance,
              (const std::vector<std::array<int, 4>>& raw_segs,
               const std::vector<bool>& flip_mask)) {
    auto segs = build_segments<double>(raw_segs, 0.25);
    const auto baseline = bentley_ottmann(segs);

    std::vector<LineSegment<double>> flipped = segs;
    for (std::size_t i = 0; i < flipped.size() && i < flip_mask.size(); ++i) {
        if (flip_mask[i]) {
            std::swap(flipped[i].start, flipped[i].end);
        }
    }
    const auto flipped_res = bentley_ottmann(flipped);
    RC_ASSERT(flipped_res == baseline);
    RC_ASSERT(count_intersections(flipped) == baseline.size());

    std::vector<LineSegment<double>> reversed(segs.rbegin(), segs.rend());
    const auto rev_res = bentley_ottmann(reversed);
    RC_ASSERT(rev_res.size() == baseline.size());

    std::vector<std::pair<std::size_t, std::size_t>> mapped_pairs;
    mapped_pairs.reserve(rev_res.size());
    const std::size_t n = segs.size();
    for (const auto& r : rev_res) {
        const std::size_t orig_a = n - 1 - r.segment1_index;
        const std::size_t orig_b = n - 1 - r.segment2_index;
        mapped_pairs.emplace_back(std::min(orig_a, orig_b), std::max(orig_a, orig_b));
    }
    std::sort(mapped_pairs.begin(), mapped_pairs.end());

    std::vector<std::pair<std::size_t, std::size_t>> base_pairs;
    base_pairs.reserve(baseline.size());
    for (const auto& r : baseline) {
        base_pairs.emplace_back(r.segment1_index, r.segment2_index);
    }
    RC_ASSERT(mapped_pairs == base_pairs);
}

RC_GTEST_PROP(BentleyOttmannPBT, FloatCoordinatesMatchBruteForceAndCount,
              (const std::vector<std::array<int, 4>>& raw_segs)) {
    const auto segs = build_segments<float>(raw_segs, 0.5f);
    assert_matches_oracle<float>(segs);
}
