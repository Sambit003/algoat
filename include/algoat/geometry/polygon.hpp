/**
 * @file polygon.hpp
 * @brief SIMD-Vectorized Point-in-Polygon (PIP) via Ray Casting & Hormann-Agathos Winding Number.
 */

#pragma once

#include "algoat/geometry/lbvh.hpp"
#include "algoat/geometry/predicates.hpp"

#if !defined(ALGOAT_SIMD_PROFILER_SCALAR_ONLY) && !defined(ALGOAT_POLYGON_SCALAR_ONLY)
#include <hwy/highway.h>
#define ALGOAT_POLYGON_HAS_HWY 1
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace algoat::geometry {

/**
 * @brief Classification of point containment within a polygon.
 */
enum class PolygonInclusion : int8_t { Outside = 0, Inside = 1, OnBoundary = 2 };

/**
 * @brief Point-in-polygon topological inclusion rule.
 */
enum class InclusionRule : uint8_t {
    EvenOdd,       // Standard Ray-Casting Parity Rule
    NonZeroWinding // Hormann-Agathos Winding Number Rule
};

inline constexpr double kPolygonBoundaryEps = 1e-10;

namespace detail {

/**
 * @brief Tests whether query point (qx, qy) lies collinear and within the bounding box of segment
 * (p1, p2).
 */
[[nodiscard]] inline bool is_point_on_segment(double qx, double qy, double x1, double y1, double x2,
                                              double y2, double edge_length) noexcept {
    const double det = (x1 - qx) * (y2 - qy) - (x2 - qx) * (y1 - qy);
    const double tol = kPolygonBoundaryEps * std::max(1.0, edge_length);
    if (std::abs(det) > tol) {
        return false;
    }

    const double seg_min_x = std::min(x1, x2) - kPolygonBoundaryEps;
    const double seg_max_x = std::max(x1, x2) + kPolygonBoundaryEps;
    const double seg_min_y = std::min(y1, y2) - kPolygonBoundaryEps;
    const double seg_max_y = std::max(y1, y2) + kPolygonBoundaryEps;

    return (qx >= seg_min_x && qx <= seg_max_x && qy >= seg_min_y && qy <= seg_max_y);
}

} // namespace detail

/**
 * @class Polygon
 * @brief SIMD-vectorized, branchless Point-in-Polygon (PIP) classification engine.
 */
class Polygon {
public:
    Polygon() = default;

    /**
     * @brief Constructs polygon from ordered vertices.
     * @param vertices Input vertices in consecutive boundary order.
     */
    explicit Polygon(std::span<const Point2D> vertices) {
        if (vertices.size() < 3) {
            return;
        }

        std::size_t n = vertices.size();
        if (n >= 4 && vertices.front().x == vertices.back().x &&
            vertices.front().y == vertices.back().y) {
            --n;
        }

        if (n < 3) {
            return;
        }

        edge_x1_.resize(n);
        edge_y1_.resize(n);
        edge_x2_.resize(n);
        edge_y2_.resize(n);
        inv_slope_.resize(n);
        edge_len_.resize(n);

        min_x_ = vertices[0].x;
        max_x_ = vertices[0].x;
        min_y_ = vertices[0].y;
        max_y_ = vertices[0].y;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& p1 = vertices[i];
            const auto& p2 = vertices[(i + 1 == n) ? 0 : i + 1];

            edge_x1_[i] = p1.x;
            edge_y1_[i] = p1.y;
            edge_x2_[i] = p2.x;
            edge_y2_[i] = p2.y;

            const double dy = p2.y - p1.y;
            const double dx = p2.x - p1.x;
            inv_slope_[i] = (dy != 0.0) ? (dx / dy) : 0.0;
            edge_len_[i] = std::hypot(dx, dy);

            min_x_ = std::min(min_x_, p1.x);
            max_x_ = std::max(max_x_, p1.x);
            min_y_ = std::min(min_y_, p1.y);
            max_y_ = std::max(max_y_, p1.y);
        }

        bbox_.min_x =
            std::nextafter(static_cast<float>(min_x_), -std::numeric_limits<float>::infinity());
        bbox_.min_y =
            std::nextafter(static_cast<float>(min_y_), -std::numeric_limits<float>::infinity());
        bbox_.min_z = 0.0f;
        bbox_._pad0 = 0.0f;
        bbox_.max_x =
            std::nextafter(static_cast<float>(max_x_), std::numeric_limits<float>::infinity());
        bbox_.max_y =
            std::nextafter(static_cast<float>(max_y_), std::numeric_limits<float>::infinity());
        bbox_.max_z = 0.0f;
        bbox_._pad1 = 0.0f;
    }

    /**
     * @brief Single point containment test.
     */
    [[nodiscard]] PolygonInclusion
    contains(const Point2D& point,
             InclusionRule rule = InclusionRule::NonZeroWinding) const noexcept {
        const std::size_t n = edge_x1_.size();
        if (n < 3) {
            return PolygonInclusion::Outside;
        }

        if (point.x < min_x_ - kPolygonBoundaryEps || point.x > max_x_ + kPolygonBoundaryEps ||
            point.y < min_y_ - kPolygonBoundaryEps || point.y > max_y_ + kPolygonBoundaryEps) {
            return PolygonInclusion::Outside;
        }

        PolygonInclusion result = PolygonInclusion::Outside;
        const Point2D pt = point;
        batch_contains(std::span<const Point2D>(&pt, 1), std::span<PolygonInclusion>(&result, 1),
                       rule);
        return result;
    }

    /**
     * @brief Scalar reference single point containment test.
     */
    [[nodiscard]] PolygonInclusion
    contains_scalar(const Point2D& point,
                    InclusionRule rule = InclusionRule::NonZeroWinding) const noexcept {
        const std::size_t n = edge_x1_.size();
        if (n < 3) {
            return PolygonInclusion::Outside;
        }

        if (point.x < min_x_ - kPolygonBoundaryEps || point.x > max_x_ + kPolygonBoundaryEps ||
            point.y < min_y_ - kPolygonBoundaryEps || point.y > max_y_ + kPolygonBoundaryEps) {
            return PolygonInclusion::Outside;
        }

        if (rule == InclusionRule::NonZeroWinding) {
            return contains_nonzero_winding_scalar(point);
        } else {
            return contains_even_odd_scalar(point);
        }
    }

    /**
     * @brief Batch query: classifies thousands of points in high-throughput SIMD stream.
     */
    void batch_contains(std::span<const Point2D> query_points,
                        std::span<PolygonInclusion> results_out,
                        InclusionRule rule = InclusionRule::NonZeroWinding) const {
        const std::size_t num_points = std::min(query_points.size(), results_out.size());
        if (num_points == 0) {
            return;
        }

        const std::size_t num_edges = edge_x1_.size();
        if (num_edges < 3) {
            for (std::size_t i = 0; i < num_points; ++i) {
                results_out[i] = PolygonInclusion::Outside;
            }
            return;
        }

        const double min_x_tol = min_x_ - kPolygonBoundaryEps;
        const double max_x_tol = max_x_ + kPolygonBoundaryEps;
        const double min_y_tol = min_y_ - kPolygonBoundaryEps;
        const double max_y_tol = max_y_ + kPolygonBoundaryEps;

#if defined(ALGOAT_POLYGON_HAS_HWY)
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<double> d;
        const std::size_t N = hn::Lanes(d);

        const auto v_min_x = hn::Set(d, min_x_tol);
        const auto v_max_x = hn::Set(d, max_x_tol);
        const auto v_min_y = hn::Set(d, min_y_tol);
        const auto v_max_y = hn::Set(d, max_y_tol);
        const auto v_eps = hn::Set(d, kPolygonBoundaryEps);

        alignas(64) double px_buf[64];
        alignas(64) double py_buf[64];

        std::size_t p_idx = 0;
        for (; p_idx + N <= num_points; p_idx += N) {
            for (std::size_t lane = 0; lane < N; ++lane) {
                px_buf[lane] = query_points[p_idx + lane].x;
                py_buf[lane] = query_points[p_idx + lane].y;
            }

            const auto v_px = hn::Load(d, px_buf);
            const auto v_py = hn::Load(d, py_buf);

            // Bounding box rejection filter
            const auto in_bbox = hn::And(hn::And(hn::Ge(v_px, v_min_x), hn::Le(v_px, v_max_x)),
                                         hn::And(hn::Ge(v_py, v_min_y), hn::Le(v_py, v_max_y)));

            if (hn::AllFalse(d, in_bbox)) {
                for (std::size_t lane = 0; lane < N; ++lane) {
                    results_out[p_idx + lane] = PolygonInclusion::Outside;
                }
                continue;
            }

            if (rule == InclusionRule::EvenOdd) {
                auto v_inside = hn::MaskFalse(d);
                auto v_near_boundary = hn::MaskFalse(d);

                for (std::size_t e = 0; e < num_edges; ++e) {
                    const auto y1 = hn::Set(d, edge_y1_[e]);
                    const auto y2 = hn::Set(d, edge_y2_[e]);
                    const auto dy = hn::Sub(v_py, y1);

                    const auto cond_y = hn::Xor(hn::Gt(y1, v_py), hn::Gt(y2, v_py));
                    const auto near_y1 = hn::Le(hn::Abs(dy), v_eps);
                    const auto near_y2 = hn::Le(hn::Abs(hn::Sub(v_py, y2)), v_eps);
                    const auto active = hn::Or(cond_y, hn::Or(near_y1, near_y2));

                    if (HWY_LIKELY(hn::AllFalse(d, active))) {
                        continue;
                    }

                    const auto x1 = hn::Set(d, edge_x1_[e]);
                    const auto inv_s = hn::Set(d, inv_slope_[e]);
                    const auto x_inter = hn::MulAdd(dy, inv_s, x1);
                    const auto crossed = hn::And(cond_y, hn::Gt(x_inter, v_px));
                    v_inside = hn::Xor(v_inside, crossed);

                    const auto diff_x = hn::Abs(hn::Sub(x_inter, v_px));
                    const auto elen = hn::Set(d, std::max(1.0, edge_len_[e]));
                    const auto tol = hn::Mul(v_eps, elen);
                    const auto near_x = hn::And(cond_y, hn::Le(diff_x, tol));
                    v_near_boundary =
                        hn::Or(v_near_boundary, hn::Or(near_x, hn::Or(near_y1, near_y2)));
                }

                v_inside = hn::And(v_inside, in_bbox);

                alignas(64) uint8_t inside_bits[64] = {0};
                alignas(64) uint8_t boundary_bits[64] = {0};
                alignas(64) uint8_t bbox_bits[64] = {0};
                hn::StoreMaskBits(d, v_inside, inside_bits);
                hn::StoreMaskBits(d, v_near_boundary, boundary_bits);
                hn::StoreMaskBits(d, in_bbox, bbox_bits);

                for (std::size_t lane = 0; lane < N; ++lane) {
                    const std::size_t byte_idx = lane / 8;
                    const std::size_t bit_idx = lane % 8;
                    const bool in_b = (bbox_bits[byte_idx] >> bit_idx) & 1;
                    if (!in_b) {
                        results_out[p_idx + lane] = PolygonInclusion::Outside;
                        continue;
                    }

                    const bool near_b = (boundary_bits[byte_idx] >> bit_idx) & 1;
                    if (HWY_UNLIKELY(near_b)) {
                        results_out[p_idx + lane] =
                            contains_even_odd_scalar(query_points[p_idx + lane]);
                    } else {
                        const bool in_p = (inside_bits[byte_idx] >> bit_idx) & 1;
                        results_out[p_idx + lane] =
                            in_p ? PolygonInclusion::Inside : PolygonInclusion::Outside;
                    }
                }
            } else {
                // NonZeroWinding rule
                auto v_winding = hn::Zero(d);
                const auto v_one = hn::Set(d, 1.0);
                auto v_near_boundary = hn::MaskFalse(d);

                for (std::size_t e = 0; e < num_edges; ++e) {
                    const auto y1 = hn::Set(d, edge_y1_[e]);
                    const auto y2 = hn::Set(d, edge_y2_[e]);
                    const auto dy = hn::Sub(v_py, y1);

                    const auto upward = hn::And(hn::Le(y1, v_py), hn::Gt(y2, v_py));
                    const auto downward = hn::And(hn::Gt(y1, v_py), hn::Le(y2, v_py));
                    const auto straddle = hn::Or(upward, downward);

                    const auto near_y1 = hn::Le(hn::Abs(dy), v_eps);
                    const auto near_y2 = hn::Le(hn::Abs(hn::Sub(v_py, y2)), v_eps);
                    const auto active = hn::Or(straddle, hn::Or(near_y1, near_y2));

                    if (HWY_LIKELY(hn::AllFalse(d, active))) {
                        continue;
                    }

                    const auto x1 = hn::Set(d, edge_x1_[e]);
                    const auto inv_s = hn::Set(d, inv_slope_[e]);
                    const auto x_inter = hn::MulAdd(dy, inv_s, x1);
                    const auto crossed = hn::And(straddle, hn::Gt(x_inter, v_px));

                    v_winding =
                        hn::Add(v_winding, hn::IfThenElseZero(hn::And(upward, crossed), v_one));
                    v_winding =
                        hn::Sub(v_winding, hn::IfThenElseZero(hn::And(downward, crossed), v_one));

                    const auto diff_x = hn::Abs(hn::Sub(x_inter, v_px));
                    const auto elen = hn::Set(d, std::max(1.0, edge_len_[e]));
                    const auto tol = hn::Mul(v_eps, elen);
                    const auto near_x = hn::And(straddle, hn::Le(diff_x, tol));
                    v_near_boundary =
                        hn::Or(v_near_boundary, hn::Or(near_x, hn::Or(near_y1, near_y2)));
                }

                const auto zero = hn::Zero(d);
                const auto v_inside = hn::And(hn::Ne(v_winding, zero), in_bbox);

                alignas(64) uint8_t inside_bits[64] = {0};
                alignas(64) uint8_t boundary_bits[64] = {0};
                alignas(64) uint8_t bbox_bits[64] = {0};
                hn::StoreMaskBits(d, v_inside, inside_bits);
                hn::StoreMaskBits(d, v_near_boundary, boundary_bits);
                hn::StoreMaskBits(d, in_bbox, bbox_bits);

                for (std::size_t lane = 0; lane < N; ++lane) {
                    const std::size_t byte_idx = lane / 8;
                    const std::size_t bit_idx = lane % 8;
                    const bool in_b = (bbox_bits[byte_idx] >> bit_idx) & 1;
                    if (!in_b) {
                        results_out[p_idx + lane] = PolygonInclusion::Outside;
                        continue;
                    }

                    const bool near_b = (boundary_bits[byte_idx] >> bit_idx) & 1;
                    if (HWY_UNLIKELY(near_b)) {
                        results_out[p_idx + lane] =
                            contains_nonzero_winding_scalar(query_points[p_idx + lane]);
                    } else {
                        const bool in_p = (inside_bits[byte_idx] >> bit_idx) & 1;
                        results_out[p_idx + lane] =
                            in_p ? PolygonInclusion::Inside : PolygonInclusion::Outside;
                    }
                }
            }
        }

        // Remainder tail points
        for (; p_idx < num_points; ++p_idx) {
            results_out[p_idx] = contains_scalar(query_points[p_idx], rule);
        }
#else
        for (std::size_t i = 0; i < num_points; ++i) {
            results_out[i] = contains_scalar(query_points[i], rule);
        }
#endif
    }

    [[nodiscard]] const AABB& bounding_box() const noexcept {
        return bbox_;
    }

    [[nodiscard]] std::size_t edge_count() const noexcept {
        return edge_x1_.size();
    }

private:
    alignas(64) std::vector<double> edge_x1_;
    alignas(64) std::vector<double> edge_y1_;
    alignas(64) std::vector<double> edge_x2_;
    alignas(64) std::vector<double> edge_y2_;
    alignas(64) std::vector<double> inv_slope_;
    alignas(64) std::vector<double> edge_len_;
    AABB bbox_{};
    double min_x_{0.0};
    double max_x_{0.0};
    double min_y_{0.0};
    double max_y_{0.0};

    [[nodiscard]] PolygonInclusion
    contains_nonzero_winding_scalar(const Point2D& point) const noexcept {
        const std::size_t n = edge_x1_.size();
        int32_t winding_number = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const double x1 = edge_x1_[i];
            const double y1 = edge_y1_[i];
            const double x2 = edge_x2_[i];
            const double y2 = edge_y2_[i];

            // Collinear and bounding box segment check
            if (detail::is_point_on_segment(point.x, point.y, x1, y1, x2, y2, edge_len_[i])) {
                return PolygonInclusion::OnBoundary;
            }

            // Hormann-Agathos upward / downward half-open ray crossing
            const bool upward = (y1 <= point.y && y2 > point.y);
            const bool downward = (y1 > point.y && y2 <= point.y);
            if (upward || downward) {
                const double x_inter = (point.y - y1) * inv_slope_[i] + x1;
                if (x_inter > point.x) {
                    winding_number += upward ? 1 : -1;
                }
            }
        }

        return (winding_number != 0) ? PolygonInclusion::Inside : PolygonInclusion::Outside;
    }

    [[nodiscard]] PolygonInclusion contains_even_odd_scalar(const Point2D& point) const noexcept {
        const std::size_t n = edge_x1_.size();
        uint8_t inside = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const double x1 = edge_x1_[i];
            const double y1 = edge_y1_[i];
            const double x2 = edge_x2_[i];
            const double y2 = edge_y2_[i];

            if (detail::is_point_on_segment(point.x, point.y, x1, y1, x2, y2, edge_len_[i])) {
                return PolygonInclusion::OnBoundary;
            }

            // Franklin branchless PNPOLY ray crossing
            const bool cond_y = (y1 > point.y) != (y2 > point.y);
            const double x_inter = (point.y - y1) * inv_slope_[i] + x1;
            inside ^= static_cast<uint8_t>(cond_y & (point.x < x_inter));
        }

        return (inside != 0) ? PolygonInclusion::Inside : PolygonInclusion::Outside;
    }
};

} // namespace algoat::geometry
