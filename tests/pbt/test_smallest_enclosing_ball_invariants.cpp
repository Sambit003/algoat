/**
 * @file test_smallest_enclosing_ball_invariants.cpp
 * @brief RapidCheck Property-Based Tests for Welzl's smallest enclosing sphere/circle.
 */

#include "algoat/geometry/smallest_enclosing_ball.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <span>
#include <vector>

using namespace algoat::geometry;

namespace {

template <std::size_t Dim>
[[nodiscard]] double dist_sq(const std::array<double, Dim>& a,
                             const std::array<double, Dim>& b) noexcept {
    double sum = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        const double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

template <std::size_t Dim>
[[nodiscard]] bool covers_all(const std::array<double, Dim>& c, double r2,
                              std::span<const std::array<double, Dim>> pts) noexcept {
    const double tol = r2 * (1.0 + 1e-11) + 1e-22;
    for (const auto& p : pts) {
        if (dist_sq<Dim>(p, c) > tol) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] BoundingSphere<double, 2>
brute_force_2d(std::span<const std::array<double, 2>> pts) noexcept {
    const std::size_t n = pts.size();
    BoundingSphere<double, 2> best{{}, std::numeric_limits<double>::infinity()};
    double best_r2 = std::numeric_limits<double>::infinity();

    auto consider = [&](const std::array<double, 2>& c, double r2) noexcept {
        if (r2 >= best_r2 || !std::isfinite(r2)) {
            return;
        }
        if (covers_all<2>(c, r2, pts)) {
            best_r2 = r2;
            best.center = c;
            best.radius = std::sqrt(std::max(0.0, r2));
        }
    };

    for (std::size_t i = 0; i < n; ++i) {
        consider(pts[i], 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            const std::array<double, 2> c2{0.5 * (pts[i][0] + pts[j][0]),
                                           0.5 * (pts[i][1] + pts[j][1])};
            consider(c2, dist_sq<2>(pts[i], c2));
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
                consider(c3, dist_sq<2>(pts[i], c3));
            }
        }
    }
    return best;
}

[[nodiscard]] BoundingSphere<double, 3>
brute_force_3d(std::span<const std::array<double, 3>> pts) noexcept {
    const std::size_t n = pts.size();
    BoundingSphere<double, 3> best{{}, std::numeric_limits<double>::infinity()};
    double best_r2 = std::numeric_limits<double>::infinity();

    auto consider = [&](const std::array<double, 3>& c, double r2) noexcept {
        if (r2 >= best_r2 || !std::isfinite(r2)) {
            return;
        }
        if (covers_all<3>(c, r2, pts)) {
            best_r2 = r2;
            best.center = c;
            best.radius = std::sqrt(std::max(0.0, r2));
        }
    };

    for (std::size_t i = 0; i < n; ++i) {
        consider(pts[i], 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            const std::array<double, 3> c2{0.5 * (pts[i][0] + pts[j][0]),
                                           0.5 * (pts[i][1] + pts[j][1]),
                                           0.5 * (pts[i][2] + pts[j][2])};
            consider(c2, dist_sq<3>(pts[i], c2));
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
                    consider(c3, dist_sq<3>(pts[i], c3));
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
                    consider(c4, dist_sq<3>(pts[i], c4));
                }
            }
        }
    }
    return best;
}

} // namespace

RC_GTEST_PROP(SmallestEnclosingBallPBT, MatchesBruteForceOracle2D,
              (const std::vector<std::array<int, 2>>& raw_pts)) {
    const std::size_t n = std::min<std::size_t>(raw_pts.size(), 25);
    RC_PRE(n >= 1);
    std::vector<std::array<double, 2>> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = {static_cast<double>(raw_pts[i][0] % 50) * 0.25,
                  static_cast<double>(raw_pts[i][1] % 50) * 0.25};
    }

    const auto got = smallest_enclosing_circle(std::span<const std::array<double, 2>>{pts});
    const auto expected = brute_force_2d(pts);
    for (const auto& p : pts) {
        RC_ASSERT(got.contains(p, 1e-9));
    }
    RC_ASSERT(std::abs(got.radius - expected.radius) <= 1e-9);
    RC_ASSERT(std::abs(got.center[0] - expected.center[0]) <= 1e-8);
    RC_ASSERT(std::abs(got.center[1] - expected.center[1]) <= 1e-8);
}

RC_GTEST_PROP(SmallestEnclosingBallPBT, MatchesBruteForceOracle3D,
              (const std::vector<std::array<int, 3>>& raw_pts)) {
    const std::size_t n = std::min<std::size_t>(raw_pts.size(), 16);
    RC_PRE(n >= 1);
    std::vector<std::array<double, 3>> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = {static_cast<double>(raw_pts[i][0] % 20) * 0.5,
                  static_cast<double>(raw_pts[i][1] % 20) * 0.5,
                  static_cast<double>(raw_pts[i][2] % 20) * 0.5};
    }

    const auto got = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
    const auto expected = brute_force_3d(pts);
    for (const auto& p : pts) {
        RC_ASSERT(got.contains(p, 1e-9));
    }
    RC_ASSERT(std::abs(got.radius - expected.radius) <= 1e-9);
    RC_ASSERT(std::abs(got.center[0] - expected.center[0]) <= 1e-7);
    RC_ASSERT(std::abs(got.center[1] - expected.center[1]) <= 1e-7);
    RC_ASSERT(std::abs(got.center[2] - expected.center[2]) <= 1e-7);
}

RC_GTEST_PROP(SmallestEnclosingBallPBT, PermutationTranslationAndScalingInvariance,
              (const std::vector<std::array<int16_t, 3>>& raw_pts, uint64_t seed, int8_t dx_raw,
               int8_t dy_raw, int8_t dz_raw, uint8_t scale_raw)) {
    const std::size_t n = std::min<std::size_t>(raw_pts.size(), 250);
    RC_PRE(n >= 1);
    std::vector<std::array<double, 3>> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = {static_cast<double>(raw_pts[i][0]) * 0.1,
                  static_cast<double>(raw_pts[i][1]) * 0.1,
                  static_cast<double>(raw_pts[i][2]) * 0.1};
    }

    const auto base = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
    for (const auto& p : pts) {
        RC_ASSERT(base.contains(p, 1e-9));
    }

    const double dx = static_cast<double>(dx_raw) * 1.5;
    const double dy = static_cast<double>(dy_raw) * 1.5;
    const double dz = static_cast<double>(dz_raw) * 1.5;
    const double s = 0.25 * static_cast<double>((scale_raw % 16) + 1);

    std::vector<std::array<double, 3>> transformed(n);
    for (std::size_t i = 0; i < n; ++i) {
        transformed[i] = {s * pts[i][0] + dx, s * pts[i][1] + dy, s * pts[i][2] + dz};
    }
    std::mt19937_64 rng(seed);
    std::shuffle(transformed.begin(), transformed.end(), rng);

    const auto got = smallest_enclosing_ball(std::span<const std::array<double, 3>>{transformed});
    const double tol = 1e-7 * std::max(1.0, got.radius);
    RC_ASSERT(std::abs(got.radius - s * base.radius) <= tol);
    RC_ASSERT(std::abs(got.center[0] - (s * base.center[0] + dx)) <= tol);
    RC_ASSERT(std::abs(got.center[1] - (s * base.center[1] + dy)) <= tol);
    RC_ASSERT(std::abs(got.center[2] - (s * base.center[2] + dz)) <= tol);
}

RC_GTEST_PROP(SmallestEnclosingBallPBT, InteriorPointRedundancy,
              (const std::vector<std::array<int8_t, 3>>& raw_pts)) {
    const std::size_t n = std::min<std::size_t>(raw_pts.size(), 150);
    RC_PRE(n >= 2);
    std::vector<std::array<double, 3>> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = {static_cast<double>(raw_pts[i][0]), static_cast<double>(raw_pts[i][1]),
                  static_cast<double>(raw_pts[i][2])};
    }

    const auto base = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
    // Add center and half-radius interior points; ball must remain invariant
    std::vector<std::array<double, 3>> augmented = pts;
    augmented.push_back(base.center);
    for (std::size_t i = 0; i < std::min<std::size_t>(n, 10); ++i) {
        augmented.push_back({0.5 * (pts[i][0] + base.center[0]), 0.5 * (pts[i][1] + base.center[1]),
                             0.5 * (pts[i][2] + base.center[2])});
    }

    const auto after = smallest_enclosing_ball(std::span<const std::array<double, 3>>{augmented});
    const double tol = 1e-9 * std::max(1.0, base.radius);
    RC_ASSERT(std::abs(after.radius - base.radius) <= tol);
    RC_ASSERT(std::abs(after.center[0] - base.center[0]) <= tol);
    RC_ASSERT(std::abs(after.center[1] - base.center[1]) <= tol);
    RC_ASSERT(std::abs(after.center[2] - base.center[2]) <= tol);
}

RC_GTEST_PROP(SmallestEnclosingBallPBT, FloatPrecisionContainmentAndOracle,
              (const std::vector<std::array<int8_t, 3>>& raw_pts)) {
    const std::size_t n = std::min<std::size_t>(raw_pts.size(), 16);
    RC_PRE(n >= 1);
    std::vector<std::array<float, 3>> pts_f(n);
    std::vector<std::array<double, 3>> pts_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(raw_pts[i][0] % 12) * 0.5f;
        const float y = static_cast<float>(raw_pts[i][1] % 12) * 0.5f;
        const float z = static_cast<float>(raw_pts[i][2] % 12) * 0.5f;
        pts_f[i] = {x, y, z};
        pts_d[i] = {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
    }

    const auto got_f = smallest_enclosing_ball(std::span<const std::array<float, 3>>{pts_f});
    const auto exp_d = brute_force_3d(pts_d);
    for (const auto& p : pts_f) {
        RC_ASSERT(got_f.contains(p, 1e-5f));
    }
    RC_ASSERT(std::abs(static_cast<double>(got_f.radius) - exp_d.radius) <= 1e-4);
}
