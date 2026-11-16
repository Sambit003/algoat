/**
 * @file smallest_enclosing_ball.hpp
 * @brief Welzl's expected O(N) Smallest Enclosing Ball (circle in 2D, sphere in 3D).
 *
 * Implements Emo Welzl's (1991) randomized linear-time algorithm combined with
 * Bernd Gaertner's (1999) translation-invariant basis representation and max-excess
 * outer pivoting, plus stack-resident move-to-front support caching for zero heap
 * allocations and strictly bounded recursion depth (<= Dim + 1 <= 4).
 *
 * References:
 * - Welzl, E. (1991). "Smallest enclosing disks (balls and ellipsoids)."
 *   New Results and New Trends in Computer Science, LNCS 555, 359-370.
 * - Gaertner, B. (1999). "Fast and robust smallest enclosing balls."
 *   Algorithms - ESA '99, LNCS 1643, 325-338.
 * - Fischer, K., Gaertner, B., & Kutz, M. (2003). "Fast smallest-enclosing-ball
 *   computation in high dimensions." Algorithms - ESA 2003, LNCS 2832, 630-641.
 */

#pragma once

#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <type_traits>

namespace algoat::geometry {

namespace detail {

inline constexpr std::size_t kSmallInputLimit = 128;
inline constexpr std::size_t kMtfCacheCapacity = 64;
inline constexpr std::size_t kSeedSampleCount = 32;
inline constexpr int kMaxPivotIterations = 128;
inline constexpr double kRelContainmentTol = 1e-13;
inline constexpr double kVerifyRelTol = 1e-12;
inline constexpr double kCollinearRelTol = 1e-12;
inline constexpr double kCollinearSqTol = 1e-24;
inline constexpr double kShewchukFallbackTol = 1e-8;
inline constexpr double kDuplicateRelTol = 1e-30;
inline constexpr double kFloatCoverRelTol = 1e-9;
inline constexpr double kHighCoordThreshold = 1e70;
inline constexpr double kLowCoordThreshold = 1e-70;
inline constexpr double kHighRadiusSqThreshold = 1e140;
inline constexpr int kMaxRescaleExp = 1020;

[[nodiscard]] constexpr bool is_finite_scalar(double value) noexcept {
    return (value - value) == 0.0;
}

template <typename T, std::size_t Dim>
[[nodiscard]] constexpr bool is_finite_point(const std::array<T, Dim>& point) noexcept {
    for (std::size_t d = 0; d < Dim; ++d) {
        if (!is_finite_scalar(static_cast<double>(point[d]))) {
            return false;
        }
    }
    return true;
}

template <std::size_t Dim> struct SphereD {
    std::array<double, Dim> center{};
    double radius_sq{-1.0};
};

template <std::size_t Dim> struct SupportSet {
    std::array<std::array<double, Dim>, Dim + 1> pts{};
    std::size_t size{0};

    [[nodiscard]] constexpr bool contains_duplicate(const std::array<double, Dim>& p,
                                                    double radius_sq) const noexcept {
        const double dup_tol = (radius_sq > 0.0) ? (kDuplicateRelTol * radius_sq) : 0.0;
        for (std::size_t k = 0; k < size; ++k) {
            double d2 = 0.0;
            for (std::size_t d = 0; d < Dim; ++d) {
                const double diff = p[d] - pts[k][d];
                d2 += diff * diff;
            }
            if (d2 <= dup_tol) {
                return true;
            }
        }
        return false;
    }
};

template <std::size_t Dim>
[[nodiscard]] constexpr double dist_sq_d(const std::array<double, Dim>& a,
                                         const std::array<double, Dim>& b) noexcept {
    double sum = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        const double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

template <std::size_t Dim>
[[nodiscard]] constexpr double dot_d(const std::array<double, Dim>& a,
                                     const std::array<double, Dim>& b) noexcept {
    double sum = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

[[nodiscard]] constexpr std::array<double, 3> cross_d(const std::array<double, 3>& a,
                                                      const std::array<double, 3>& b) noexcept {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template <std::size_t Dim>
[[nodiscard]] constexpr SphereD<Dim> make_diametric_d(const std::array<double, Dim>& a,
                                                      const std::array<double, Dim>& b) noexcept {
    SphereD<Dim> s;
    for (std::size_t d = 0; d < Dim; ++d) {
        s.center[d] = a[d] + 0.5 * (b[d] - a[d]);
    }
    s.radius_sq = std::max(dist_sq_d(a, s.center), dist_sq_d(b, s.center));
    return s;
}

template <std::size_t Dim>
[[nodiscard]] constexpr bool contains_padded_d(const SphereD<Dim>& s,
                                               const std::array<double, Dim>& p,
                                               double rel_tol = kRelContainmentTol) noexcept {
    if (s.radius_sq < 0.0) {
        return false;
    }
    return dist_sq_d(p, s.center) <= s.radius_sq * (1.0 + rel_tol);
}

template <std::size_t Dim>
[[nodiscard]] inline SphereD<Dim> exact_circumsphere(const SupportSet<Dim>& support) noexcept;

template <std::size_t Dim>
[[nodiscard]] inline SphereD<Dim> best_subset_sphere(const SupportSet<Dim>& support) noexcept {
    SphereD<Dim> best;
    SphereD<Dim> fallback;
    double fallback_max_d2 = std::numeric_limits<double>::infinity();

    auto consider = [&](const SphereD<Dim>& cand) noexcept {
        if (cand.radius_sq < 0.0 || !is_finite_scalar(cand.radius_sq)) {
            return;
        }
        double max_d2 = cand.radius_sq;
        bool valid = true;
        for (std::size_t i = 0; i < support.size; ++i) {
            const double d2 = dist_sq_d(support.pts[i], cand.center);
            if (d2 > max_d2) {
                max_d2 = d2;
            }
            if (d2 > cand.radius_sq * (1.0 + kVerifyRelTol)) {
                valid = false;
            }
        }
        if (valid && (best.radius_sq < 0.0 || cand.radius_sq < best.radius_sq)) {
            best = cand;
        }
        if (max_d2 < fallback_max_d2) {
            fallback_max_d2 = max_d2;
            fallback = cand;
            fallback.radius_sq = max_d2;
        }
    };

    for (std::size_t i = 0; i < support.size; ++i) {
        consider(SphereD<Dim>{support.pts[i], 0.0});
    }

    for (std::size_t i = 0; i < support.size; ++i) {
        for (std::size_t j = i + 1; j < support.size; ++j) {
            consider(make_diametric_d(support.pts[i], support.pts[j]));
        }
    }

    if constexpr (Dim == 3) {
        if (support.size == 4) {
            for (std::size_t skip = 0; skip < 4; ++skip) {
                SupportSet<3> sub;
                for (std::size_t i = 0; i < 4; ++i) {
                    if (i != skip) {
                        sub.pts[sub.size++] = support.pts[i];
                    }
                }
                consider(exact_circumsphere<3>(sub));
            }
        }
    }

    return (best.radius_sq >= 0.0) ? best : fallback;
}

template <std::size_t Dim>
[[nodiscard]] inline SphereD<Dim> exact_circumsphere(const SupportSet<Dim>& support) noexcept {
    if (support.size == 0) {
        return SphereD<Dim>{{}, -1.0};
    }
    if (support.size == 1) {
        return SphereD<Dim>{support.pts[0], 0.0};
    }
    if (support.size == 2) {
        return make_diametric_d(support.pts[0], support.pts[1]);
    }

    const auto& r = support.pts;
    if constexpr (Dim == 2) {
        // 2D 3-point circumcircle via shifted Cramer's determinant rule relative to r[0]
        const std::array<double, 2> u{r[1][0] - r[0][0], r[1][1] - r[0][1]};
        const std::array<double, 2> v{r[2][0] - r[0][0], r[2][1] - r[0][1]};
        const double u2 = u[0] * u[0] + u[1] * u[1];
        const double v2 = v[0] * v[0] + v[1] * v[1];
        const double det = 2.0 * orient2d_raw(r[0].data(), r[1].data(), r[2].data());
        const double u_len = std::sqrt(u2);
        const double v_len = std::sqrt(v2);
        if (u_len == 0.0 || v_len == 0.0 || (std::abs(det) / u_len) <= kCollinearRelTol * v_len ||
            !is_finite_scalar(det)) {
            return best_subset_sphere<2>(support);
        }
        const double dx = (u2 * v[1] - v2 * u[1]) / det;
        const double dy = (u[0] * v2 - v[0] * u2) / det;
        const std::array<double, 2> center{r[0][0] + dx, r[0][1] + dy};
        if (!is_finite_point(center)) {
            return best_subset_sphere<2>(support);
        }
        const double r2 =
            std::max({dist_sq_d(r[0], center), dist_sq_d(r[1], center), dist_sq_d(r[2], center)});
        return SphereD<2>{center, r2};
    } else {
        static_assert(Dim == 3, "exact_circumsphere specializes 2D and 3D only.");
        if (support.size == 3) {
            // 3D 3-point circumcircle in aff(r[0], r[1], r[2])
            const std::array<double, 3> u{r[1][0] - r[0][0], r[1][1] - r[0][1], r[1][2] - r[0][2]};
            const std::array<double, 3> v{r[2][0] - r[0][0], r[2][1] - r[0][1], r[2][2] - r[0][2]};
            const double u2 = dot_d(u, u);
            const double v2 = dot_d(v, v);
            const double uv = dot_d(u, v);
            const auto cr = cross_d(u, v);
            const double denom = 2.0 * dot_d(cr, cr);
            if (u2 == 0.0 || v2 == 0.0 || (denom / u2) <= kCollinearSqTol * v2 ||
                !is_finite_scalar(denom)) {
                return best_subset_sphere<3>(support);
            }
            const double alpha = v2 * (u2 - uv) / denom;
            const double beta = u2 * (v2 - uv) / denom;
            const std::array<double, 3> center{r[0][0] + alpha * u[0] + beta * v[0],
                                               r[0][1] + alpha * u[1] + beta * v[1],
                                               r[0][2] + alpha * u[2] + beta * v[2]};
            if (!is_finite_point(center)) {
                return best_subset_sphere<3>(support);
            }
            const double r2 = std::max(
                {dist_sq_d(r[0], center), dist_sq_d(r[1], center), dist_sq_d(r[2], center)});
            return SphereD<3>{center, r2};
        }

        // 3D 4-point circumsphere via shifted Cramer's determinant rule relative to r[0]
        const std::array<double, 3> u{r[1][0] - r[0][0], r[1][1] - r[0][1], r[1][2] - r[0][2]};
        const std::array<double, 3> v{r[2][0] - r[0][0], r[2][1] - r[0][1], r[2][2] - r[0][2]};
        const std::array<double, 3> w{r[3][0] - r[0][0], r[3][1] - r[0][1], r[3][2] - r[0][2]};
        const auto c_vw = cross_d(v, w);
        const auto c_wu = cross_d(w, u);
        const auto c_uv = cross_d(u, v);
        const double u2 = dot_d(u, u);
        const double v2 = dot_d(v, v);
        const double w2 = dot_d(w, w);
        const double u_len = std::sqrt(u2);
        const double v_len = std::sqrt(v2);
        const double w_len = std::sqrt(w2);
        if (u_len == 0.0 || v_len == 0.0 || w_len == 0.0) {
            return best_subset_sphere<3>(support);
        }
        double det = 2.0 * dot_d(u, c_vw);
        if (!is_finite_scalar(det)) {
            return best_subset_sphere<3>(support);
        }
        double rel_det = (std::abs(det) / u_len) / v_len;
        if (rel_det <= kShewchukFallbackTol * w_len) {
            det = -2.0 * orient3d_raw(r[0].data(), r[1].data(), r[2].data(), r[3].data());
            if (!is_finite_scalar(det)) {
                return best_subset_sphere<3>(support);
            }
            rel_det = (std::abs(det) / u_len) / v_len;
        }
        if (rel_det <= kCollinearRelTol * w_len || det == 0.0) {
            return best_subset_sphere<3>(support);
        }
        const std::array<double, 3> center{
            r[0][0] + (u2 * c_vw[0] + v2 * c_wu[0] + w2 * c_uv[0]) / det,
            r[0][1] + (u2 * c_vw[1] + v2 * c_wu[1] + w2 * c_uv[1]) / det,
            r[0][2] + (u2 * c_vw[2] + v2 * c_wu[2] + w2 * c_uv[2]) / det};
        if (!is_finite_point(center)) {
            return best_subset_sphere<3>(support);
        }
        const double r2 = std::max({dist_sq_d(r[0], center), dist_sq_d(r[1], center),
                                    dist_sq_d(r[2], center), dist_sq_d(r[3], center)});
        return SphereD<3>{center, r2};
    }
}

template <std::size_t Dim, std::size_t Cap> struct MtfBuffer {
    std::array<std::array<double, Dim>, Cap> pts{};
    std::size_t size{0};

    [[nodiscard]] constexpr std::span<std::array<double, Dim>> active() noexcept {
        return std::span<std::array<double, Dim>>{pts.data(), size};
    }

    constexpr void push_front(const std::array<double, Dim>& pt) noexcept {
        std::size_t pos = size;
        for (std::size_t k = 0; k < size; ++k) {
            if (dist_sq_d(pts[k], pt) == 0.0) {
                pos = k;
                break;
            }
        }
        if (pos == size) {
            if (size < Cap) {
                pos = size++;
            } else {
                pos = Cap - 1;
            }
        }
        for (std::size_t k = pos; k > 0; --k) {
            pts[k] = pts[k - 1];
        }
        pts[0] = pt;
    }
};

template <std::size_t Dim>
[[nodiscard]] inline SphereD<Dim> welzl_mtf(std::span<std::array<double, Dim>> buf,
                                            std::size_t limit, SupportSet<Dim>& support) noexcept {
    SphereD<Dim> ball = exact_circumsphere<Dim>(support);
    if (support.size == Dim + 1) {
        return ball;
    }

    for (std::size_t i = 0; i < limit; ++i) {
        const auto& p = buf[i];
        if (!contains_padded_d(ball, p) && !support.contains_duplicate(p, ball.radius_sq)) {
            const std::array<double, Dim> pt = p;
            support.pts[support.size++] = pt;
            ball = welzl_mtf<Dim>(buf, i, support);
            --support.size;
            for (std::size_t k = i; k > 0; --k) {
                buf[k] = buf[k - 1];
            }
            buf[0] = pt;
        }
    }

    // At root level, verify no strictly smaller subset of the front support points encloses buf
    if (support.size == 0 && limit >= 2) {
        const std::size_t cand_n = std::min<std::size_t>(limit, Dim + 1);
        std::array<std::array<double, Dim>, Dim + 1> front{};
        for (std::size_t i = 0; i < cand_n; ++i) {
            front[i] = buf[i];
        }
        auto try_candidate = [&](const SphereD<Dim>& cand) noexcept {
            if (cand.radius_sq < 0.0 || cand.radius_sq >= ball.radius_sq) {
                return;
            }
            for (std::size_t i = 0; i < limit; ++i) {
                if (!contains_padded_d(cand, buf[i])) {
                    return;
                }
            }
            ball = cand;
        };
        for (std::size_t i = 0; i < cand_n; ++i) {
            for (std::size_t j = i + 1; j < cand_n; ++j) {
                try_candidate(make_diametric_d(front[i], front[j]));
            }
        }
        if constexpr (Dim == 3) {
            if (cand_n >= 3) {
                for (std::size_t i = 0; i < cand_n; ++i) {
                    for (std::size_t j = i + 1; j < cand_n; ++j) {
                        for (std::size_t k = j + 1; k < cand_n; ++k) {
                            SupportSet<3> tri{{front[i], front[j], front[k]}, 3};
                            try_candidate(exact_circumsphere<3>(tri));
                        }
                    }
                }
            }
        }
    }

    return ball;
}

template <typename T, std::size_t Dim>
[[nodiscard]] constexpr std::array<double, Dim> to_double_point(const std::array<T, Dim>& p,
                                                                double scale = 1.0) noexcept {
    std::array<double, Dim> out{};
    for (std::size_t d = 0; d < Dim; ++d) {
        out[d] = static_cast<double>(p[d]) * scale;
    }
    return out;
}

template <typename T, std::size_t Dim>
[[nodiscard]] constexpr double scaled_dist_sq(const std::array<T, Dim>& p, double scale,
                                              const std::array<double, Dim>& c) noexcept {
    double sum = 0.0;
    for (std::size_t d = 0; d < Dim; ++d) {
        const double diff = static_cast<double>(p[d]) * scale - c[d];
        sum += diff * diff;
    }
    return sum;
}

template <typename T, std::size_t Dim>
[[nodiscard]] inline int compute_rescaling_exp(std::span<const std::array<T, Dim>> points,
                                               double& out_max_abs) noexcept {
    double max_abs = 0.0;
    for (const auto& pt : points) {
        if (!is_finite_point(pt)) {
            continue;
        }
        for (std::size_t d = 0; d < Dim; ++d) {
            const double v = std::abs(static_cast<double>(pt[d]));
            if (v > max_abs) {
                max_abs = v;
            }
        }
    }
    out_max_abs = max_abs;
    if (max_abs == 0.0) {
        return 0;
    }
    return std::clamp(-std::ilogb(max_abs), -kMaxRescaleExp, kMaxRescaleExp);
}

template <typename T> struct PointArrayTraits : std::false_type {};

template <typename T, std::size_t Dim>
struct PointArrayTraits<std::array<T, Dim>> : std::true_type {
    using value_type = T;
    static constexpr std::size_t dim = Dim;
};

template <typename R>
concept IsPointArrayRange =
    std::ranges::contiguous_range<R> &&
    PointArrayTraits<std::remove_cvref_t<std::ranges::range_value_t<R>>>::value &&
    std::is_floating_point_v<typename PointArrayTraits<
        std::remove_cvref_t<std::ranges::range_value_t<R>>>::value_type> &&
    (PointArrayTraits<std::remove_cvref_t<std::ranges::range_value_t<R>>>::dim == 2 ||
     PointArrayTraits<std::remove_cvref_t<std::ranges::range_value_t<R>>>::dim == 3);

template <typename R>
concept IsPoint2DArrayRange =
    IsPointArrayRange<R> &&
    (PointArrayTraits<std::remove_cvref_t<std::ranges::range_value_t<R>>>::dim == 2);

} // namespace detail

/**
 * @struct BoundingSphere
 * @brief Euclidean bounding sphere (disk in 2D, ball in 3D) in @p Dim dimensions.
 *
 * @tparam T   Floating-point coordinate and radius type (default @c double).
 * @tparam Dim Spatial dimensionality (2 or 3, default @c 2).
 */
template <typename T = double, std::size_t Dim = 2> struct BoundingSphere {
    static_assert(std::is_floating_point_v<T>,
                  "BoundingSphere requires a floating-point coordinate type.");
    static_assert(Dim == 2 || Dim == 3, "BoundingSphere supports Dim == 2 or Dim == 3.");

    std::array<T, Dim> center{};
    T radius{};

    /**
     * @brief Tests whether @p point lies on or inside this bounding sphere within @p epsilon.
     *
     * Negative @p radius or non-finite coordinates always return @c false.
     */
    [[nodiscard]] constexpr bool contains(const std::array<T, Dim>& point,
                                          T epsilon = static_cast<T>(1e-9)) const noexcept {
        if (!(radius >= T(0))) {
            return false;
        }
        const double r = static_cast<double>(radius);
        const double eps = static_cast<double>(epsilon);
        if (!detail::is_finite_scalar(r) || !detail::is_finite_scalar(eps)) {
            return false;
        }

        const double half_bound = 0.5 * r + 0.5 * eps;
        if (half_bound < 0.0) {
            return false;
        }

        double max_half_diff = 0.0;
        std::array<double, Dim> half_diffs{};
        for (std::size_t d = 0; d < Dim; ++d) {
            const double p = static_cast<double>(point[d]);
            const double c = static_cast<double>(center[d]);
            if (!detail::is_finite_scalar(p) || !detail::is_finite_scalar(c)) {
                return false;
            }
            const double hd = 0.5 * p - 0.5 * c;
            const double ahd = hd < 0.0 ? -hd : hd;
            if (ahd > max_half_diff) {
                max_half_diff = ahd;
            }
            half_diffs[d] = hd;
        }

        if (max_half_diff > half_bound) {
            return false;
        }
        if (max_half_diff == 0.0) {
            return true;
        }

        constexpr double kScaleUp = 1e140;
        constexpr double kScaleDown = 1e-140;
        if (max_half_diff > kScaleUp || half_bound > kScaleUp || max_half_diff < kScaleDown) {
            const double inv = 1.0 / max_half_diff;
            double scaled_dist_sq = 0.0;
            for (std::size_t d = 0; d < Dim; ++d) {
                const double s = half_diffs[d] * inv;
                scaled_dist_sq += s * s;
            }
            const double scaled_bound = half_bound * inv;
            return scaled_dist_sq <= scaled_bound * scaled_bound;
        }

        double dist_sq = 0.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            const double diff = static_cast<double>(point[d]) - static_cast<double>(center[d]);
            dist_sq += diff * diff;
        }
        const double bound = r + eps;
        return dist_sq <= bound * bound;
    }

    /**
     * @brief Computes the area (2D) or volume (3D) enclosed by this sphere.
     *
     * Returns @c T(0) for non-positive or non-finite radius.
     */
    [[nodiscard]] constexpr T volume() const noexcept {
        if (!(radius > T(0))) {
            return T(0);
        }
        const double r = static_cast<double>(radius);
        if (!detail::is_finite_scalar(r)) {
            return T(0);
        }
        if constexpr (Dim == 2) {
            return static_cast<T>(std::numbers::pi * r * r);
        } else {
            return static_cast<T>((4.0 / 3.0) * std::numbers::pi * r * r * r);
        }
    }
};

namespace detail {

template <typename T, std::size_t Dim>
[[nodiscard]] inline bool solve_small_welzl(std::span<const std::array<T, Dim>> points,
                                            double scale, bool allow_rescale,
                                            SphereD<Dim>& out_ball, bool& out_empty) noexcept {
    MtfBuffer<Dim, kSmallInputLimit> small_buf{};
    double max_coord = 0.0;
    for (const auto& pt : points) {
        if (is_finite_point(pt)) {
            const auto dpt = to_double_point(pt, scale);
            for (std::size_t d = 0; d < Dim; ++d) {
                max_coord = std::max(max_coord, std::abs(dpt[d]));
            }
            small_buf.pts[small_buf.size++] = dpt;
        }
    }
    if (small_buf.size == 0) {
        out_empty = true;
        return true;
    }
    out_empty = false;
    if (allow_rescale &&
        (max_coord > kHighCoordThreshold || (max_coord > 0.0 && max_coord < kLowCoordThreshold))) {
        return false;
    }

    // Deterministic Knuth-hash Fisher-Yates shuffle on stack buffer
    for (std::size_t i = small_buf.size - 1; i > 0; --i) {
        const std::size_t j =
            static_cast<std::size_t>((i * 2654435761ULL + 1013904223ULL) % (i + 1));
        std::swap(small_buf.pts[i], small_buf.pts[j]);
    }

    SupportSet<Dim> support{};
    out_ball = welzl_mtf<Dim>(small_buf.active(), small_buf.size, support);
    if (allow_rescale &&
        (!is_finite_scalar(out_ball.radius_sq) || out_ball.radius_sq > kHighRadiusSqThreshold)) {
        return false;
    }

    double max_d2 = out_ball.radius_sq;
    for (std::size_t i = 0; i < small_buf.size; ++i) {
        max_d2 = std::max(max_d2, dist_sq_d(small_buf.pts[i], out_ball.center));
    }
    if (max_d2 > out_ball.radius_sq && max_d2 <= out_ball.radius_sq * (1.0 + kFloatCoverRelTol)) {
        out_ball.radius_sq = max_d2;
    }
    return true;
}

template <typename T, std::size_t Dim, typename ViolationFn>
[[nodiscard]] inline bool scan_points_fast(std::span<const std::array<T, Dim>> points, double scale,
                                           SphereD<Dim>& ball, double& r_sq_tol,
                                           ViolationFn&& on_violation) noexcept {
    const std::size_t n = points.size();
    std::array<double, Dim> c = ball.center;
    std::size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        std::array<double, 4> d2{};
        for (std::size_t k = 0; k < 4; ++k) {
            d2[k] = scaled_dist_sq(points[i + k], scale, c);
        }
        if (d2[0] > r_sq_tol || d2[1] > r_sq_tol || d2[2] > r_sq_tol || d2[3] > r_sq_tol)
            [[unlikely]] {
            for (std::size_t k = 0; k < 4; ++k) {
                if (scaled_dist_sq(points[i + k], scale, ball.center) > r_sq_tol) {
                    if (on_violation(i + k)) {
                        return false;
                    }
                    c = ball.center;
                }
            }
        }
    }
    for (; i < n; ++i) {
        if (scaled_dist_sq(points[i], scale, c) > r_sq_tol) [[unlikely]] {
            if (on_violation(i)) {
                return false;
            }
            c = ball.center;
        }
    }
    return true;
}

template <typename T, std::size_t Dim>
[[nodiscard]] inline BoundingSphere<T, Dim>
finalize_bounding_sphere(std::span<const std::array<T, Dim>> points, double scale, double inv_scale,
                         const SphereD<Dim>& ball) noexcept {
    BoundingSphere<T, Dim> out{};
    for (std::size_t d = 0; d < Dim; ++d) {
        out.center[d] = static_cast<T>(ball.center[d] * inv_scale);
    }
    const double exact_r = std::sqrt(std::max(0.0, ball.radius_sq)) * inv_scale;
    out.radius = static_cast<T>(exact_r);

    if constexpr (sizeof(T) < sizeof(double)) {
        // Cover all finite input points despite T-precision rounding of center and radius
        double max_d2 = 0.0;
        for (const auto& pt : points) {
            if (!is_finite_point(pt)) {
                continue;
            }
            double d2 = 0.0;
            for (std::size_t d = 0; d < Dim; ++d) {
                const double diff = static_cast<double>(pt[d]) - static_cast<double>(out.center[d]);
                d2 += diff * diff;
            }
            if (d2 > max_d2) {
                max_d2 = d2;
            }
        }
        const T min_r = static_cast<T>(std::sqrt(max_d2));
        if (out.radius < min_r) {
            out.radius = min_r;
        }
        if (static_cast<double>(out.radius) * static_cast<double>(out.radius) < max_d2) {
            out.radius = std::nextafter(out.radius, std::numeric_limits<T>::infinity());
        }
    } else if (ball.radius_sq > 0.0 && scale == 1.0 && (out.radius * out.radius) < ball.radius_sq) {
        // Guard against 0.5-ULP downward rounding of std::sqrt(ball.radius_sq)
        out.radius = std::nextafter(out.radius, std::numeric_limits<T>::infinity());
    }

    return out;
}

template <typename T, std::size_t Dim>
[[nodiscard]] inline BoundingSphere<T, Dim>
smallest_enclosing_ball_impl(std::span<const std::array<T, Dim>> points, int scale_exp) noexcept {
    const std::size_t n = points.size();
    if (n == 0) {
        return BoundingSphere<T, Dim>{{}, T(0)};
    }

    const bool allow_rescale = (scale_exp == 0);
    const double scale = allow_rescale ? 1.0 : std::ldexp(1.0, scale_exp);
    const double inv_scale = allow_rescale ? 1.0 : std::ldexp(1.0, -scale_exp);

    auto retry_rescaled = [&]() noexcept -> BoundingSphere<T, Dim> {
        double max_abs = 0.0;
        const int exp = compute_rescaling_exp(points, max_abs);
        if (max_abs == 0.0) {
            return BoundingSphere<T, Dim>{{}, T(0)};
        }
        return smallest_enclosing_ball_impl<T, Dim>(points, (exp != 0) ? exp : 1);
    };

    if (n <= kSmallInputLimit) {
        SphereD<Dim> ball;
        bool is_empty = false;
        if (!solve_small_welzl(points, scale, allow_rescale, ball, is_empty)) {
            return retry_rescaled();
        }
        if (is_empty) {
            return BoundingSphere<T, Dim>{{}, T(0)};
        }
        return finalize_bounding_sphere(points, scale, inv_scale, ball);
    }

    MtfBuffer<Dim, kMtfCacheCapacity> mtf_buf{};
    double seed_max_coord = 0.0;
    for (std::size_t s = 0; s < kSeedSampleCount; ++s) {
        const std::size_t idx = (s * n) / kSeedSampleCount;
        if (is_finite_point(points[idx])) {
            const auto pt = to_double_point(points[idx], scale);
            for (std::size_t d = 0; d < Dim; ++d) {
                seed_max_coord = std::max(seed_max_coord, std::abs(pt[d]));
            }
            mtf_buf.pts[mtf_buf.size++] = pt;
        }
    }
    if (allow_rescale && (seed_max_coord > kHighCoordThreshold ||
                          (seed_max_coord > 0.0 && seed_max_coord < kLowCoordThreshold))) {
        return retry_rescaled();
    }

    SphereD<Dim> ball;
    SupportSet<Dim> support{};
    if (mtf_buf.size > 0) {
        ball = welzl_mtf<Dim>(mtf_buf.active(), mtf_buf.size, support);
        if (allow_rescale &&
            (!is_finite_scalar(ball.radius_sq) || ball.radius_sq > kHighRadiusSqThreshold)) {
            return retry_rescaled();
        }
    }

    std::size_t last_update_idx = n;
    double r_sq_tol = (ball.radius_sq < 0.0) ? -1.0 : (ball.radius_sq * (1.0 + kVerifyRelTol));

    auto on_violation = [&](std::size_t i) noexcept -> bool {
        if (!is_finite_point(points[i])) {
            return false;
        }
        const auto pt = to_double_point(points[i], scale);
        if (allow_rescale) {
            for (std::size_t d = 0; d < Dim; ++d) {
                if (std::abs(pt[d]) > kHighCoordThreshold) {
                    return true;
                }
            }
        }
        last_update_idx = i;
        support.pts[0] = pt;
        support.size = 1;
        ball = welzl_mtf<Dim>(mtf_buf.active(), mtf_buf.size, support);
        support.size = 0;
        mtf_buf.push_front(pt);
        if (allow_rescale &&
            (!is_finite_scalar(ball.radius_sq) || ball.radius_sq > kHighRadiusSqThreshold)) {
            return true;
        }
        r_sq_tol = ball.radius_sq * (1.0 + kVerifyRelTol);
        return false;
    };

    if (!scan_points_fast(points, scale, ball, r_sq_tol, on_violation)) {
        return retry_rescaled();
    }

    if (ball.radius_sq <= 0.0 && allow_rescale) {
        // Recover tiny non-zero clouds if strided samples missed sub-underflow points
        double max_abs = 0.0;
        const int exp = compute_rescaling_exp(points, max_abs);
        if (max_abs == 0.0) {
            return BoundingSphere<T, Dim>{{}, T(0)};
        }
        if (max_abs < kLowCoordThreshold && exp != 0) {
            return smallest_enclosing_ball_impl<T, Dim>(points, exp);
        }
    }

    if (ball.radius_sq < 0.0) {
        return BoundingSphere<T, Dim>{{}, T(0)};
    }

    // Verify unchecked prefix [0, last_update_idx) and pivot via Gaertner max-excess rule
    std::size_t check_limit = (last_update_idx < n) ? last_update_idx : 0;
    if (last_update_idx == n && mtf_buf.size < kSeedSampleCount) {
        check_limit = n;
    }

    int no_growth_streak = 0;
    for (int iter = 0; iter < kMaxPivotIterations && check_limit > 0; ++iter) {
        double max_d2 = r_sq_tol;
        std::size_t max_idx = n;
        for (std::size_t j = 0; j < check_limit; ++j) {
            const double d2 = scaled_dist_sq(points[j], scale, ball.center);
            if (d2 > max_d2 && is_finite_point(points[j])) {
                max_d2 = d2;
                max_idx = j;
            }
        }
        if (max_idx == n) {
            break;
        }
        const SphereD<Dim> old_ball = ball;
        const auto pt = to_double_point(points[max_idx], scale);
        support.pts[0] = pt;
        support.size = 1;
        ball = welzl_mtf<Dim>(mtf_buf.active(), mtf_buf.size, support);
        support.size = 0;
        mtf_buf.push_front(pt);
        if (ball.radius_sq <= old_ball.radius_sq) {
            ball = old_ball;
            ++no_growth_streak;
            if (max_d2 <= ball.radius_sq * (1.0 + kFloatCoverRelTol) || no_growth_streak >= 2) {
                ball.radius_sq = std::max(ball.radius_sq, max_d2);
                break;
            }
        } else {
            no_growth_streak = 0;
        }
        r_sq_tol = ball.radius_sq * (1.0 + kVerifyRelTol);
        check_limit = n;
    }

    return finalize_bounding_sphere(points, scale, inv_scale, ball);
}

} // namespace detail

/**
 * @brief Computes the exact smallest enclosing disk/ball in expected O(N) time.
 *
 * Uses Welzl's (1991) randomized/move-to-front algorithm with shifted Cramer
 * determinant circumspheres and Gaertner's (1999) max-excess outer verification.
 * Executes with zero heap allocations and recursion depth bounded by @c Dim + 1 <= 4.
 *
 * @tparam T   Floating-point coordinate type (default @c double).
 * @tparam Dim Spatial dimensionality (2 or 3, default @c 2).
 * @param points Read-only contiguous span of @p Dim -dimensional points.
 * @return Exact smallest @c BoundingSphere enclosing all finite input points.
 */
template <typename T = double, std::size_t Dim = 2>
[[nodiscard]] BoundingSphere<T, Dim>
smallest_enclosing_ball(std::span<const std::array<T, Dim>> points) noexcept {
    static_assert(std::is_floating_point_v<T>,
                  "smallest_enclosing_ball requires a floating-point coordinate type.");
    static_assert(
        Dim == 2 || Dim == 3,
        "smallest_enclosing_ball specializes 2D and 3D point sets (Dim == 2 or Dim == 3).");
    return detail::smallest_enclosing_ball_impl<T, Dim>(points, 0);
}

/**
 * @brief Mutable span overload for @c smallest_enclosing_ball.
 */
template <typename T, std::size_t Dim>
[[nodiscard]] BoundingSphere<T, Dim>
smallest_enclosing_ball(std::span<std::array<T, Dim>> points) noexcept {
    return smallest_enclosing_ball<T, Dim>(
        std::span<const std::array<T, Dim>>{points.data(), points.size()});
}

/**
 * @brief Contiguous range convenience overload for @c smallest_enclosing_ball.
 */
template <std::ranges::contiguous_range R>
    requires detail::IsPointArrayRange<R>
[[nodiscard]] auto smallest_enclosing_ball(R&& points) noexcept {
    using PointType = std::remove_cvref_t<std::ranges::range_value_t<R>>;
    using T = typename detail::PointArrayTraits<PointType>::value_type;
    constexpr std::size_t Dim = detail::PointArrayTraits<PointType>::dim;
    return smallest_enclosing_ball<T, Dim>(
        std::span<const std::array<T, Dim>>{std::ranges::data(points), std::ranges::size(points)});
}

/**
 * @brief 2D specialization convenience alias returning the smallest enclosing circle.
 */
template <typename T = double>
[[nodiscard]] BoundingSphere<T, 2>
smallest_enclosing_circle(std::span<const std::array<T, 2>> points) noexcept {
    return smallest_enclosing_ball<T, 2>(points);
}

/**
 * @brief Mutable span overload for @c smallest_enclosing_circle.
 */
template <typename T>
[[nodiscard]] BoundingSphere<T, 2>
smallest_enclosing_circle(std::span<std::array<T, 2>> points) noexcept {
    return smallest_enclosing_ball<T, 2>(
        std::span<const std::array<T, 2>>{points.data(), points.size()});
}

/**
 * @brief Contiguous range convenience overload for @c smallest_enclosing_circle.
 */
template <std::ranges::contiguous_range R>
    requires detail::IsPoint2DArrayRange<R>
[[nodiscard]] auto smallest_enclosing_circle(R&& points) noexcept {
    using PointType = std::remove_cvref_t<std::ranges::range_value_t<R>>;
    using T = typename detail::PointArrayTraits<PointType>::value_type;
    return smallest_enclosing_ball<T, 2>(
        std::span<const std::array<T, 2>>{std::ranges::data(points), std::ranges::size(points)});
}

} // namespace algoat::geometry
