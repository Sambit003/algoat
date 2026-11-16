/**
 * @file diff_smallest_enclosing_ball_fuzzer.cpp
 * @brief Differential LibFuzzer harness for Welzl's smallest enclosing sphere under ASan+UBSan.
 */

#include "algoat/geometry/smallest_enclosing_ball.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
    const double tol = r2 * (1.0 + 1e-10) + 1e-22;
    for (const auto& p : pts) {
        if (dist_sq<Dim>(p, c) > tol) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] double brute_force_radius_2d(std::span<const std::array<double, 2>> pts) noexcept {
    const std::size_t n = pts.size();
    double best_r2 = std::numeric_limits<double>::infinity();
    auto consider = [&](const std::array<double, 2>& c, double r2) noexcept {
        if (r2 < best_r2 && std::isfinite(r2) && covers_all<2>(c, r2, pts)) {
            best_r2 = r2;
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
    return std::isfinite(best_r2) ? std::sqrt(std::max(0.0, best_r2)) : 0.0;
}

[[nodiscard]] double brute_force_radius_3d(std::span<const std::array<double, 3>> pts) noexcept {
    const std::size_t n = pts.size();
    double best_r2 = std::numeric_limits<double>::infinity();
    auto consider = [&](const std::array<double, 3>& c, double r2) noexcept {
        if (r2 < best_r2 && std::isfinite(r2) && covers_all<3>(c, r2, pts)) {
            best_r2 = r2;
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
    return std::isfinite(best_r2) ? std::sqrt(std::max(0.0, best_r2)) : 0.0;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 1 + 3 * sizeof(float)) {
        return 0;
    }
    const uint8_t mode = data[0];
    const uint8_t* payload = data + 1;
    const std::size_t payload_bytes = size - 1;

    if ((mode & 1u) == 0) {
        // 2D grid or arbitrary float points
        if ((mode & 2u) != 0) {
            const std::size_t count = std::min<std::size_t>(payload_bytes / 2, 24);
            std::vector<std::array<double, 2>> pts(count);
            for (std::size_t i = 0; i < count; ++i) {
                pts[i] = {static_cast<double>(static_cast<int8_t>(payload[2 * i]) % 7),
                          static_cast<double>(static_cast<int8_t>(payload[2 * i + 1]) % 7)};
            }
            const auto got = smallest_enclosing_circle(std::span<const std::array<double, 2>>{pts});
            if (!std::isfinite(got.radius) || got.radius < 0.0) {
                __builtin_trap();
            }
            for (const auto& p : pts) {
                if (!got.contains(p, 1e-9)) {
                    __builtin_trap();
                }
            }
            if (!pts.empty()) {
                const double exp_r = brute_force_radius_2d(pts);
                if (std::abs(got.radius - exp_r) > 1e-8 * std::max(1.0, exp_r)) {
                    __builtin_trap();
                }
            }
        } else {
            const std::size_t count =
                std::min<std::size_t>(payload_bytes / (2 * sizeof(float)), 160);
            std::vector<std::array<float, 2>> pts_f(count);
            std::vector<std::array<double, 2>> pts_d(count);
            for (std::size_t i = 0; i < count; ++i) {
                std::array<float, 2> raw{};
                std::memcpy(raw.data(), payload + i * 2 * sizeof(float), 2 * sizeof(float));
                pts_f[i] = raw;
                pts_d[i] = {static_cast<double>(raw[0]), static_cast<double>(raw[1])};
            }
            const auto got_f =
                smallest_enclosing_circle(std::span<const std::array<float, 2>>{pts_f});
            const auto got_d =
                smallest_enclosing_circle(std::span<const std::array<double, 2>>{pts_d});
            if (!std::isfinite(got_d.radius) || got_d.radius < 0.0 || got_f.radius < 0.0) {
                __builtin_trap();
            }
            if (got_d.radius <= static_cast<double>(std::numeric_limits<float>::max()) * 0.99999 &&
                !std::isfinite(got_f.radius)) {
                __builtin_trap();
            }
            const double tol = 1e-9 * std::max(1.0, got_d.radius);
            for (const auto& p : pts_d) {
                if (std::isfinite(p[0]) && std::isfinite(p[1]) && !got_d.contains(p, tol)) {
                    __builtin_trap();
                }
            }
        }
        return 0;
    }

    // 3D grid or arbitrary float points
    if ((mode & 2u) != 0) {
        const std::size_t count = std::min<std::size_t>(payload_bytes / 3, 18);
        std::vector<std::array<double, 3>> pts(count);
        for (std::size_t i = 0; i < count; ++i) {
            pts[i] = {static_cast<double>(static_cast<int8_t>(payload[3 * i]) % 5),
                      static_cast<double>(static_cast<int8_t>(payload[3 * i + 1]) % 5),
                      static_cast<double>(static_cast<int8_t>(payload[3 * i + 2]) % 5)};
        }
        const auto got = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts});
        if (!std::isfinite(got.radius) || got.radius < 0.0) {
            __builtin_trap();
        }
        for (const auto& p : pts) {
            if (!got.contains(p, 1e-9)) {
                __builtin_trap();
            }
        }
        if (!pts.empty()) {
            const double exp_r = brute_force_radius_3d(pts);
            if (std::abs(got.radius - exp_r) > 1e-8 * std::max(1.0, exp_r)) {
                __builtin_trap();
            }
        }
    } else {
        const std::size_t count = std::min<std::size_t>(payload_bytes / (3 * sizeof(float)), 160);
        std::vector<std::array<float, 3>> pts_f(count);
        std::vector<std::array<double, 3>> pts_d(count);
        for (std::size_t i = 0; i < count; ++i) {
            std::array<float, 3> raw{};
            std::memcpy(raw.data(), payload + i * 3 * sizeof(float), 3 * sizeof(float));
            pts_f[i] = raw;
            pts_d[i] = {static_cast<double>(raw[0]), static_cast<double>(raw[1]),
                        static_cast<double>(raw[2])};
        }
        const auto got_f = smallest_enclosing_ball(std::span<const std::array<float, 3>>{pts_f});
        const auto got_d = smallest_enclosing_ball(std::span<const std::array<double, 3>>{pts_d});
        if (!std::isfinite(got_d.radius) || got_d.radius < 0.0 || got_f.radius < 0.0) {
            __builtin_trap();
        }
        if (got_d.radius <= static_cast<double>(std::numeric_limits<float>::max()) * 0.99999 &&
            !std::isfinite(got_f.radius)) {
            __builtin_trap();
        }
        const double tol = 1e-9 * std::max(1.0, got_d.radius);
        for (const auto& p : pts_d) {
            if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
                !got_d.contains(p, tol)) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
