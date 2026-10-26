/**
 * @file convex_hull.hpp
 * @brief 2D and 3D Convex Hull suite implementing Andrew's Monotone Chain,
 * Chan's output-sensitive algorithm, and 3D Quickhull.
 *
 * Provides industrial-grade, robust, and zero-allocation (where applicable)
 * convex hull generation for planar and spatial point clouds using exact
 * geometric predicates (Shewchuk's orient2d and orient3d).
 */

#pragma once

#include "algoat/geometry/predicates.hpp"
#include "algoat/sorting/pdqsort.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::geometry {

struct Polygon2D {
    std::vector<Point2D> vertices; // Counter-clockwise ordered vertices
    double area{0.0};
    double perimeter{0.0};
};

struct Polyhedron3D {
    std::vector<Point3D> vertices;
    std::vector<std::array<uint32_t, 3>> triangle_facets;
    double volume{0.0};
    double surface_area{0.0};
};

namespace detail {

// ============================================================================
// 2D Vector & Metrics Helpers
// ============================================================================

[[nodiscard]] inline double polygon_perimeter(std::span<const Point2D> vertices) noexcept {
    const std::size_t n = vertices.size();
    if (n < 2) {
        return 0.0;
    }
    if (n == 2) {
        return 2.0 * std::hypot(vertices[1].x - vertices[0].x, vertices[1].y - vertices[0].y);
    }
    double sum = 0.0;
    double compensation = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t next = (i + 1 == n) ? 0 : (i + 1);
        const double edge =
            std::hypot(vertices[next].x - vertices[i].x, vertices[next].y - vertices[i].y);
        const double t = sum + edge;
        if (std::abs(sum) >= std::abs(edge)) {
            compensation += (sum - t) + edge;
        } else {
            compensation += (edge - t) + sum;
        }
        sum = t;
    }
    return sum + compensation;
}

[[nodiscard]] inline double polygon_area(std::span<const Point2D> vertices) noexcept {
    const std::size_t n = vertices.size();
    if (n < 3) {
        return 0.0;
    }
    const double x0 = vertices[0].x;
    const double y0 = vertices[0].y;
    double sum = 0.0;
    double compensation = 0.0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double xi = vertices[i].x - x0;
        const double yi = vertices[i].y - y0;
        const double xi1 = vertices[i + 1].x - x0;
        const double yi1 = vertices[i + 1].y - y0;
        const double cross = std::fma(xi, yi1, -xi1 * yi);
        const double t = sum + cross;
        if (std::abs(sum) >= std::abs(cross)) {
            compensation += (sum - t) + cross;
        } else {
            compensation += (cross - t) + sum;
        }
        sum = t;
    }
    return std::abs(sum + compensation) * 0.5;
}

// Bisection to find tangent vertex from point p to counter-clockwise convex polygon
[[nodiscard]] inline Point2D find_tangent(const Point2D& query_pt,
                                          std::span<const Point2D> polygon) noexcept {
    const std::size_t n = polygon.size();
    if (n == 0) {
        return query_pt;
    }
    if (n == 1) {
        return polygon[0];
    }

    // Direct linear scan for small polygons
    if (n <= 16) {
        Point2D best = polygon[0];
        double best_dist_sq = (best.x - query_pt.x) * (best.x - query_pt.x) +
                              (best.y - query_pt.y) * (best.y - query_pt.y);
        for (std::size_t i = 1; i < n; ++i) {
            const Orientation orient = orient2d(query_pt, best, polygon[i]);
            if (orient == Orientation::Clockwise) {
                best = polygon[i];
                best_dist_sq = (best.x - query_pt.x) * (best.x - query_pt.x) +
                               (best.y - query_pt.y) * (best.y - query_pt.y);
            } else if (orient == Orientation::Collinear) {
                const double d_sq = (polygon[i].x - query_pt.x) * (polygon[i].x - query_pt.x) +
                                    (polygon[i].y - query_pt.y) * (polygon[i].y - query_pt.y);
                if (d_sq > best_dist_sq) {
                    best = polygon[i];
                    best_dist_sq = d_sq;
                }
            }
        }
        return best;
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (query_pt.x == polygon[i].x && query_pt.y == polygon[i].y) {
            return polygon[(i + 1) % n];
        }
    }

    auto is_ccw = [&](std::size_t i, std::size_t j) noexcept -> bool {
        return orient2d(query_pt, polygon[i], polygon[j]) == Orientation::CounterClockwise;
    };
    auto is_cw = [&](std::size_t i, std::size_t j) noexcept -> bool {
        return orient2d(query_pt, polygon[i], polygon[j]) == Orientation::Clockwise;
    };

    if (is_cw(1, 0) && !is_ccw(n - 1, 0)) {
        return polygon[0];
    }

    std::size_t low = 0;
    std::size_t high = n;
    while (high - low > 1) {
        const std::size_t mid = low + (high - low) / 2;
        const std::size_t mid_next = (mid + 1 == n) ? 0 : (mid + 1);
        const std::size_t mid_prev = (mid == 0) ? (n - 1) : (mid - 1);

        const bool mid_turns_cw = is_cw(mid_next, mid);
        if (mid_turns_cw && !is_ccw(mid_prev, mid)) {
            return polygon[mid];
        }

        const std::size_t low_next = (low + 1 == n) ? 0 : (low + 1);
        const bool low_turns_ccw = is_ccw(low_next, low);

        if (low_turns_ccw) {
            if (mid_turns_cw) {
                high = mid;
            } else {
                if (is_ccw(low, mid)) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
        } else {
            if (!mid_turns_cw) {
                low = mid;
            } else {
                if (is_cw(low, mid)) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
        }
    }

    const Point2D candidate_a = polygon[low];
    const Point2D candidate_b = (high == n) ? polygon[0] : polygon[high];
    const Orientation final_orient = orient2d(query_pt, candidate_a, candidate_b);
    if (final_orient == Orientation::CounterClockwise) {
        return candidate_a;
    }
    if (final_orient == Orientation::Clockwise) {
        return candidate_b;
    }
    const double dist_a_sq = (candidate_a.x - query_pt.x) * (candidate_a.x - query_pt.x) +
                             (candidate_a.y - query_pt.y) * (candidate_a.y - query_pt.y);
    const double dist_b_sq = (candidate_b.x - query_pt.x) * (candidate_b.x - query_pt.x) +
                             (candidate_b.y - query_pt.y) * (candidate_b.y - query_pt.y);
    return dist_a_sq > dist_b_sq ? candidate_a : candidate_b;
}

// Akl-Toussaint heuristic pre-filter: eliminates points strictly inside bounding quadrilateral
template <std::ranges::random_access_range R>
[[nodiscard]] std::vector<Point2D> akl_toussaint_filter(R&& pts) {
    const std::size_t n = std::ranges::size(pts);
    if (n <= 32) {
        std::vector<Point2D> out;
        out.reserve(n);
        for (const auto& pt : pts) {
            out.push_back(pt);
        }
        return out;
    }

    std::size_t min_x_idx = 0, max_x_idx = 0, min_y_idx = 0, max_y_idx = 0;
    std::size_t idx = 0;
    for (const auto& pt : pts) {
        if (pt.x < pts[min_x_idx].x)
            min_x_idx = idx;
        if (pt.x > pts[max_x_idx].x)
            max_x_idx = idx;
        if (pt.y < pts[min_y_idx].y)
            min_y_idx = idx;
        if (pt.y > pts[max_y_idx].y)
            max_y_idx = idx;
        ++idx;
    }

    std::array<Point2D, 4> quad = {pts[min_x_idx], pts[min_y_idx], pts[max_x_idx], pts[max_y_idx]};
    if (orient2d(quad[0], quad[1], quad[2]) != Orientation::CounterClockwise) {
        std::swap(quad[1], quad[3]);
    }

    std::array<double, 4> edge_dx{};
    std::array<double, 4> edge_dy{};
    std::array<double, 4> edge_c{};
    for (std::size_t k = 0; k < 4; ++k) {
        const std::size_t next = (k + 1) % 4;
        edge_dx[k] = quad[next].x - quad[k].x;
        edge_dy[k] = quad[next].y - quad[k].y;
        edge_c[k] = edge_dx[k] * quad[k].y - edge_dy[k] * quad[k].x;
    }

    std::vector<Point2D> filtered;
    filtered.reserve(std::max<std::size_t>(n / 10, 64));

    for (const auto& pt : pts) {
        const double px = pt.x;
        const double py = pt.y;
        if (edge_dx[0] * py - edge_dy[0] * px > edge_c[0] + 1e-12 &&
            edge_dx[1] * py - edge_dy[1] * px > edge_c[1] + 1e-12 &&
            edge_dx[2] * py - edge_dy[2] * px > edge_c[2] + 1e-12 &&
            edge_dx[3] * py - edge_dy[3] * px > edge_c[3] + 1e-12) {
            continue;
        }
        filtered.push_back(pt);
    }

    if (filtered.empty()) {
        filtered.reserve(n);
        for (const auto& pt : pts) {
            filtered.push_back(pt);
        }
    }
    return filtered;
}

// ============================================================================
// 3D Vector & Exact Predicate Helpers
// ============================================================================

[[nodiscard]] constexpr Point3D sub3(const Point3D& a, const Point3D& b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] constexpr Point3D add3(const Point3D& a, const Point3D& b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] constexpr Point3D mul3(const Point3D& a, double s) noexcept {
    return {a.x * s, a.y * s, a.z * s};
}

[[nodiscard]] constexpr double dot3(const Point3D& a, const Point3D& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] constexpr Point3D cross3(const Point3D& a, const Point3D& b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] inline double norm3(const Point3D& p) noexcept {
    return std::sqrt(dot3(p, p));
}

[[nodiscard]] inline Point3D normalize3(const Point3D& p) noexcept {
    const double len = norm3(p);
    return len > 1e-15 ? mul3(p, 1.0 / len) : p;
}

} // namespace detail

// ============================================================================
// 1. Andrew's Monotone Chain 2D Convex Hull
// ============================================================================

/**
 * @brief Computes 2D Convex Hull via Andrew's Monotone Chain algorithm in O(N log N).
 *
 * Uses coordinate sorting and exact Shewchuk predicates (`orient2d`) to construct
 * the strict counter-clockwise convex polygon.
 *
 * @tparam R Random-access range of Point2D.
 * @param points Input point cloud.
 * @return Polygon2D CCW-ordered vertices with exact area and perimeter.
 */
template <std::ranges::random_access_range R>
    requires std::same_as<std::ranges::range_value_t<R>, Point2D>
[[nodiscard]] Polygon2D convex_hull_2d_andrew(R&& points) {
    const std::size_t input_size = std::ranges::size(points);
    if (input_size == 0) {
        return Polygon2D{{}, 0.0, 0.0};
    }
    if (input_size == 1) {
        return Polygon2D{std::vector<Point2D>{*std::ranges::begin(points)}, 0.0, 0.0};
    }

    std::vector<Point2D> pts;
    if constexpr (std::is_rvalue_reference_v<R&&> &&
                  std::same_as<std::remove_cvref_t<R>, std::vector<Point2D>>) {
        pts = std::forward<R>(points);
    } else {
        pts.reserve(input_size);
        for (auto&& pt : points) {
            pts.push_back(pt);
        }
    }

    // 1. Lexicographical coordinate sort
    sorting::pdqsort(pts.begin(), pts.end(), [](const Point2D& a, const Point2D& b) noexcept {
        return (a.x < b.x) | ((a.x == b.x) & (a.y < b.y));
    });

    // 2. In-place deduplication
    auto unique_end =
        std::unique(pts.begin(), pts.end(), [](const Point2D& a, const Point2D& b) noexcept {
            return a.x == b.x && a.y == b.y;
        });
    pts.erase(unique_end, pts.end());

    const std::size_t n = pts.size();
    if (n <= 1) {
        return Polygon2D{std::move(pts), 0.0, 0.0};
    }
    if (n == 2) {
        const double perim = 2.0 * std::hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
        return Polygon2D{std::move(pts), 0.0, perim};
    }

    // 3. Monotone chain sweeps
    std::vector<Point2D> hull(2 * n);
    std::size_t k = 0;

    // Lower hull
    for (std::size_t i = 0; i < n; ++i) {
        while (k >= 2 &&
               orient2d(hull[k - 2], hull[k - 1], pts[i]) != Orientation::CounterClockwise) {
            --k;
        }
        hull[k++] = pts[i];
    }

    const std::size_t lower_count = k;
    const std::size_t t = lower_count + 1;

    // Upper hull
    for (std::size_t i = n - 1; i > 0; --i) {
        const std::size_t idx = i - 1;
        while (k >= t &&
               orient2d(hull[k - 2], hull[k - 1], pts[idx]) != Orientation::CounterClockwise) {
            --k;
        }
        hull[k++] = pts[idx];
    }

    if (k > 1) {
        --k; // pop redundant closing endpoint
    }
    hull.resize(k);

    const double area = detail::polygon_area(hull);
    const double perim = detail::polygon_perimeter(hull);

    return Polygon2D{std::move(hull), area, perim};
}

// ============================================================================
// 2. Output-Sensitive Chan's 2D Convex Hull
// ============================================================================

/**
 * @brief Computes 2D Convex Hull via Chan's Output-Sensitive algorithm in O(N log H).
 *
 * Combines grouping with Andrew's monotone chain and Jarvis march tangent
 * bisections to achieve optimal output-sensitive performance when H << N.
 *
 * @tparam R Random-access range of Point2D.
 * @param points Input point cloud.
 * @return Polygon2D CCW-ordered vertices with exact area and perimeter.
 */
template <std::ranges::random_access_range R>
    requires std::same_as<std::ranges::range_value_t<R>, Point2D>
[[nodiscard]] Polygon2D convex_hull_2d_chan(R&& points) {
    const std::size_t raw_n = std::ranges::size(points);
    if (raw_n <= 3) {
        return convex_hull_2d_andrew(points);
    }

    std::vector<Point2D> pts = detail::akl_toussaint_filter(points);
    const std::size_t n = pts.size();
    if (n <= 3) {
        return convex_hull_2d_andrew(pts);
    }

    Point2D p1 = pts[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (pts[i].x < p1.x || (pts[i].x == p1.x && pts[i].y < p1.y)) {
            p1 = pts[i];
        }
    }

    for (std::size_t t = 1;; ++t) {
        std::size_t m = n;
        if (t <= 4) {
            const std::size_t p2 = 1ULL << (1ULL << t);
            if (p2 < n) {
                m = p2;
            }
        }

        if (m >= n) {
            return convex_hull_2d_andrew(pts);
        }

        const std::size_t num_chunks = (n + m - 1) / m;
        std::vector<Polygon2D> subhulls;
        subhulls.reserve(num_chunks);

        for (std::size_t i = 0; i < n; i += m) {
            const std::size_t chunk_size = std::min(m, n - i);
            std::span<const Point2D> chunk_span(&pts[i], chunk_size);
            Polygon2D sub = convex_hull_2d_andrew(chunk_span);
            if (!sub.vertices.empty()) {
                subhulls.push_back(std::move(sub));
            }
        }

        std::vector<Point2D> hull;
        hull.reserve(m + 1);
        hull.push_back(p1);

        bool closed = false;

        for (std::size_t step = 0; step < m; ++step) {
            const Point2D pk = hull.back();
            Point2D best = pk;
            bool found_cand = false;
            double best_dist_sq = 0.0;

            for (const auto& sh : subhulls) {
                const Point2D cand = detail::find_tangent(pk, sh.vertices);
                if (cand.x == pk.x && cand.y == pk.y) {
                    continue;
                }

                const double cand_dist_sq =
                    (cand.x - pk.x) * (cand.x - pk.x) + (cand.y - pk.y) * (cand.y - pk.y);
                if (!found_cand) {
                    best = cand;
                    best_dist_sq = cand_dist_sq;
                    found_cand = true;
                } else {
                    const Orientation orient = orient2d(pk, best, cand);
                    if (orient == Orientation::Clockwise) {
                        best = cand;
                        best_dist_sq = cand_dist_sq;
                    } else if (orient == Orientation::Collinear) {
                        if (cand_dist_sq > best_dist_sq) {
                            best = cand;
                            best_dist_sq = cand_dist_sq;
                        }
                    }
                }
            }

            if (!found_cand || (best.x == p1.x && best.y == p1.y)) {
                closed = true;
                break;
            }

            hull.push_back(best);
        }

        if (closed) {
            const double area = detail::polygon_area(hull);
            const double perim = detail::polygon_perimeter(hull);
            return Polygon2D{std::move(hull), area, perim};
        }
    }
}

// ============================================================================
// 3. 3D Quickhull Convex Polyhedron Generation
// ============================================================================

namespace detail {

struct QuickhullFacet {
    std::array<std::size_t, 3> v{};
    std::array<std::size_t, 3> neighbors{};
    Point3D normal{};
    double offset{0.0};
    std::vector<std::size_t> outside_set;
    std::size_t furthest_pt{0};
    double furthest_dist{0.0};
    bool visible{false};
    bool deleted{false};
};

struct HorizonEdge {
    std::size_t u{0};
    std::size_t v{0};
    std::size_t non_vis_facet{0};
    std::size_t non_vis_edge_idx{0};
};

} // namespace detail

/**
 * @brief Computes 3D Convex Hull via Barber-Dobkin-Huhdanpaa Quickhull algorithm.
 *
 * Uses Shewchuk exact orientation predicates (`orient3d`) and signed distance functions
 * to generate a robust closed 2-manifold triangular boundary mesh with outward normals.
 * Degenerate 0D, 1D, and 2D coplanar clouds are handled gracefully.
 *
 * @tparam R Random-access range of Point3D.
 * @param points Input spatial point cloud.
 * @return Polyhedron3D Vertices, triangular facets, exact volume, and surface area.
 */
template <std::ranges::random_access_range R>
    requires std::same_as<std::ranges::range_value_t<R>, Point3D>
[[nodiscard]] Polyhedron3D convex_hull_3d_quickhull(R&& points) {
    const std::size_t input_size = std::ranges::size(points);
    if (input_size == 0) {
        return Polyhedron3D{{}, {}, 0.0, 0.0};
    }
    if (input_size == 1) {
        return Polyhedron3D{std::vector<Point3D>{*std::ranges::begin(points)}, {}, 0.0, 0.0};
    }

    std::vector<Point3D> S;
    S.reserve(input_size);
    for (auto&& pt : points) {
        S.push_back(pt);
    }

    // Deduplicate identical points
    sorting::pdqsort(S.begin(), S.end(), [](const Point3D& a, const Point3D& b) noexcept {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.z < b.z;
    });
    auto it = std::unique(S.begin(), S.end(), [](const Point3D& a, const Point3D& b) noexcept {
        return std::abs(a.x - b.x) < 1e-14 && std::abs(a.y - b.y) < 1e-14 &&
               std::abs(a.z - b.z) < 1e-14;
    });
    S.erase(it, S.end());

    const std::size_t n = S.size();
    if (n <= 1) {
        return Polyhedron3D{std::move(S), {}, 0.0, 0.0};
    }
    if (n <= 3) {
        return Polyhedron3D{std::move(S), {}, 0.0, 0.0};
    }

    // 1. Find 6 extreme points along X, Y, Z
    std::size_t min_x = 0, max_x = 0, min_y = 0, max_y = 0, min_z = 0, max_z = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (S[i].x < S[min_x].x)
            min_x = i;
        if (S[i].x > S[max_x].x)
            max_x = i;
        if (S[i].y < S[min_y].y)
            min_y = i;
        if (S[i].y > S[max_y].y)
            max_y = i;
        if (S[i].z < S[min_z].z)
            min_z = i;
        if (S[i].z > S[max_z].z)
            max_z = i;
    }

    const std::array<std::size_t, 6> extremes{min_x, max_x, min_y, max_y, min_z, max_z};
    std::size_t p0 = extremes[0], p1 = extremes[1];
    double max_dist_sq = detail::dot3(detail::sub3(S[p0], S[p1]), detail::sub3(S[p0], S[p1]));
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = i + 1; j < 6; ++j) {
            const double d_sq = detail::dot3(detail::sub3(S[extremes[i]], S[extremes[j]]),
                                             detail::sub3(S[extremes[i]], S[extremes[j]]));
            if (d_sq > max_dist_sq) {
                max_dist_sq = d_sq;
                p0 = extremes[i];
                p1 = extremes[j];
            }
        }
    }

    if (max_dist_sq < 1e-14) {
        return Polyhedron3D{{S[0]}, {}, 0.0, 0.0};
    }

    // 2. Find p2 maximizing distance to line p0-p1
    const Point3D line_v = detail::sub3(S[p1], S[p0]);
    const double line_len_sq = detail::dot3(line_v, line_v);
    std::size_t p2 = p0;
    double max_line_dist_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Point3D cp = detail::cross3(detail::sub3(S[i], S[p0]), line_v);
        const double d_sq = detail::dot3(cp, cp) / line_len_sq;
        if (d_sq > max_line_dist_sq) {
            max_line_dist_sq = d_sq;
            p2 = i;
        }
    }

    if (max_line_dist_sq < 1e-14) {
        return Polyhedron3D{{S[p0], S[p1]}, {}, 0.0, 0.0};
    }

    // 3. Find p3 maximizing distance to plane (p0, p1, p2)
    const Point3D plane_normal =
        detail::cross3(detail::sub3(S[p1], S[p0]), detail::sub3(S[p2], S[p0]));
    const double plane_norm_sq = detail::dot3(plane_normal, plane_normal);
    std::size_t p3 = p0;
    double max_plane_dist_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double proj = detail::dot3(detail::sub3(S[i], S[p0]), plane_normal);
        const double d_sq = (proj * proj) / plane_norm_sq;
        if (d_sq > max_plane_dist_sq) {
            max_plane_dist_sq = d_sq;
            p3 = i;
        }
    }

    // Handle 2D coplanar degeneracy by computing 2D convex hull on projected plane
    if (max_plane_dist_sq < 1e-14) {
        const Point3D u_axis = detail::normalize3(line_v);
        const Point3D w_axis = detail::normalize3(plane_normal);
        const Point3D v_axis = detail::cross3(w_axis, u_axis);

        std::vector<Point2D> projected;
        projected.reserve(n);
        for (const auto& pt : S) {
            const Point3D d = detail::sub3(pt, S[p0]);
            projected.push_back({detail::dot3(d, u_axis), detail::dot3(d, v_axis)});
        }

        Polygon2D hull2d = convex_hull_2d_andrew(projected);
        std::vector<Point3D> hull3d;
        hull3d.reserve(hull2d.vertices.size());
        for (const auto& v : hull2d.vertices) {
            Point3D pt3 = detail::add3(
                S[p0], detail::add3(detail::mul3(u_axis, v.x), detail::mul3(v_axis, v.y)));
            hull3d.push_back(pt3);
        }
        return Polyhedron3D{std::move(hull3d), {}, 0.0, hull2d.area};
    }

    // Ensure initial tetrahedron vertices form outward-pointing faces
    const Point3D n012 = detail::cross3(detail::sub3(S[p1], S[p0]), detail::sub3(S[p2], S[p0]));
    if (detail::dot3(n012, detail::sub3(S[p3], S[p0])) > 0.0) {
        std::swap(p1, p2);
    }

    std::vector<detail::QuickhullFacet> facets;
    facets.reserve(64);

    auto make_facet = [&](std::size_t a, std::size_t b, std::size_t c) {
        detail::QuickhullFacet f;
        f.v = {a, b, c};
        const Point3D raw_n = detail::cross3(detail::sub3(S[b], S[a]), detail::sub3(S[c], S[a]));
        f.normal = detail::normalize3(raw_n);
        f.offset = -detail::dot3(f.normal, S[a]);
        return f;
    };

    facets.push_back(make_facet(p0, p1, p2)); // 0: base
    facets.push_back(make_facet(p0, p2, p3)); // 1
    facets.push_back(make_facet(p0, p3, p1)); // 2
    facets.push_back(make_facet(p1, p3, p2)); // 3

    facets[0].neighbors = {2, 3, 1};
    facets[1].neighbors = {0, 3, 2};
    facets[2].neighbors = {1, 3, 0};
    facets[3].neighbors = {2, 1, 0};

    // Partition remaining points into outside sets using orient3d and signed distance
    for (std::size_t i = 0; i < n; ++i) {
        if (i == p0 || i == p1 || i == p2 || i == p3)
            continue;

        double max_d = 1e-12;
        std::size_t best_f = 4;
        for (std::size_t f_idx = 0; f_idx < 4; ++f_idx) {
            const double d = detail::dot3(facets[f_idx].normal, S[i]) + facets[f_idx].offset;
            const Orientation o =
                orient3d(S[facets[f_idx].v[0]], S[facets[f_idx].v[1]], S[facets[f_idx].v[2]], S[i]);
            if (d > max_d || o == Orientation::Clockwise) {
                if (d > max_d) {
                    max_d = d;
                    best_f = f_idx;
                }
            }
        }
        if (best_f < 4) {
            facets[best_f].outside_set.push_back(i);
            if (max_d > facets[best_f].furthest_dist) {
                facets[best_f].furthest_dist = max_d;
                facets[best_f].furthest_pt = i;
            }
        }
    }

    // Reusable buffers to eliminate per-iteration heap reallocations
    std::vector<std::size_t> visible_facets;
    std::vector<std::size_t> bfs_queue;
    std::vector<detail::HorizonEdge> horizon;
    std::vector<std::size_t> pts_to_reassign;

    // Quickhull main loop
    while (true) {
        std::size_t eye_facet_idx = facets.size();
        double max_furthest = 1e-12;
        for (std::size_t i = 0; i < facets.size(); ++i) {
            if (!facets[i].deleted && !facets[i].outside_set.empty()) {
                if (facets[i].furthest_dist > max_furthest) {
                    max_furthest = facets[i].furthest_dist;
                    eye_facet_idx = i;
                }
            }
        }

        if (eye_facet_idx == facets.size()) {
            break; // Hull complete
        }

        const std::size_t eye_pt = facets[eye_facet_idx].furthest_pt;
        const Point3D eye_pos = S[eye_pt];

        for (auto& f : facets) {
            f.visible = false;
        }

        visible_facets.clear();
        bfs_queue.clear();
        horizon.clear();

        bfs_queue.push_back(eye_facet_idx);
        facets[eye_facet_idx].visible = true;
        visible_facets.push_back(eye_facet_idx);

        std::size_t head = 0;
        while (head < bfs_queue.size()) {
            const std::size_t curr_idx = bfs_queue[head++];
            for (std::size_t edge_idx = 0; edge_idx < 3; ++edge_idx) {
                const std::size_t neighbor_idx = facets[curr_idx].neighbors[edge_idx];
                if (facets[neighbor_idx].deleted)
                    continue;

                if (!facets[neighbor_idx].visible) {
                    const double d = detail::dot3(facets[neighbor_idx].normal, eye_pos) +
                                     facets[neighbor_idx].offset;
                    const Orientation o =
                        orient3d(S[facets[neighbor_idx].v[0]], S[facets[neighbor_idx].v[1]],
                                 S[facets[neighbor_idx].v[2]], eye_pos);
                    if (d > 1e-12 || o == Orientation::Clockwise) {
                        facets[neighbor_idx].visible = true;
                        visible_facets.push_back(neighbor_idx);
                        bfs_queue.push_back(neighbor_idx);
                    } else {
                        const std::size_t u = facets[curr_idx].v[edge_idx];
                        const std::size_t v = facets[curr_idx].v[(edge_idx + 1) % 3];
                        std::size_t n_edge = 0;
                        for (std::size_t k = 0; k < 3; ++k) {
                            if (facets[neighbor_idx].v[k] == v &&
                                facets[neighbor_idx].v[(k + 1) % 3] == u) {
                                n_edge = k;
                                break;
                            }
                        }
                        horizon.push_back(detail::HorizonEdge{u, v, neighbor_idx, n_edge});
                    }
                }
            }
        }

        if (horizon.empty()) {
            facets[eye_facet_idx].outside_set.clear();
            continue;
        }

        // Order horizon edges into oriented cycle
        for (std::size_t i = 0; i < horizon.size(); ++i) {
            for (std::size_t j = i + 1; j < horizon.size(); ++j) {
                if (horizon[j].u == horizon[i].v) {
                    std::swap(horizon[i + 1], horizon[j]);
                    break;
                }
            }
        }

        const std::size_t new_start_idx = facets.size();
        const std::size_t M = horizon.size();

        for (std::size_t m = 0; m < M; ++m) {
            facets.push_back(make_facet(horizon[m].u, horizon[m].v, eye_pt));
        }

        for (std::size_t m = 0; m < M; ++m) {
            const std::size_t curr = new_start_idx + m;
            const std::size_t next = new_start_idx + (m + 1) % M;
            const std::size_t prev = new_start_idx + (m + M - 1) % M;

            facets[curr].neighbors[0] = horizon[m].non_vis_facet;
            facets[curr].neighbors[1] = next;
            facets[curr].neighbors[2] = prev;

            facets[horizon[m].non_vis_facet].neighbors[horizon[m].non_vis_edge_idx] = curr;
        }

        pts_to_reassign.clear();
        for (std::size_t vf : visible_facets) {
            for (std::size_t pt : facets[vf].outside_set) {
                if (pt != eye_pt) {
                    pts_to_reassign.push_back(pt);
                }
            }
            facets[vf].deleted = true;
            facets[vf].outside_set.clear();
        }

        for (std::size_t pt : pts_to_reassign) {
            const Point3D p_pos = S[pt];
            double max_d = 1e-12;
            std::size_t best_f = facets.size();

            for (std::size_t m = 0; m < M; ++m) {
                const std::size_t f_idx = new_start_idx + m;
                const double d = detail::dot3(facets[f_idx].normal, p_pos) + facets[f_idx].offset;
                const Orientation o = orient3d(S[facets[f_idx].v[0]], S[facets[f_idx].v[1]],
                                               S[facets[f_idx].v[2]], p_pos);
                if (d > max_d || o == Orientation::Clockwise) {
                    if (d > max_d) {
                        max_d = d;
                        best_f = f_idx;
                    }
                }
            }

            if (best_f < facets.size()) {
                facets[best_f].outside_set.push_back(pt);
                if (max_d > facets[best_f].furthest_dist) {
                    facets[best_f].furthest_dist = max_d;
                    facets[best_f].furthest_pt = pt;
                }
            }
        }
    }

    // Build final Polyhedron3D
    std::vector<std::size_t> active_facets;
    for (std::size_t i = 0; i < facets.size(); ++i) {
        if (!facets[i].deleted) {
            active_facets.push_back(i);
        }
    }

    std::vector<std::size_t> vert_map(n, n);
    std::vector<Point3D> out_vertices;
    out_vertices.reserve(n);

    for (std::size_t fi : active_facets) {
        for (std::size_t k = 0; k < 3; ++k) {
            const std::size_t v_idx = facets[fi].v[k];
            if (vert_map[v_idx] == n) {
                vert_map[v_idx] = out_vertices.size();
                out_vertices.push_back(S[v_idx]);
            }
        }
    }

    std::vector<std::array<uint32_t, 3>> out_facets;
    out_facets.reserve(active_facets.size());

    for (std::size_t fi : active_facets) {
        out_facets.push_back({static_cast<uint32_t>(vert_map[facets[fi].v[0]]),
                              static_cast<uint32_t>(vert_map[facets[fi].v[1]]),
                              static_cast<uint32_t>(vert_map[facets[fi].v[2]])});
    }

    Point3D centroid{0.0, 0.0, 0.0};
    for (const auto& v : out_vertices) {
        centroid = detail::add3(centroid, v);
    }
    if (!out_vertices.empty()) {
        centroid = detail::mul3(centroid, 1.0 / static_cast<double>(out_vertices.size()));
    }

    double total_area = 0.0;
    double total_vol = 0.0;

    for (const auto& tri : out_facets) {
        const Point3D v0 = out_vertices[tri[0]];
        const Point3D v1 = out_vertices[tri[1]];
        const Point3D v2 = out_vertices[tri[2]];

        const Point3D cp = detail::cross3(detail::sub3(v1, v0), detail::sub3(v2, v0));
        total_area += 0.5 * detail::norm3(cp);
        total_vol += (1.0 / 6.0) * detail::dot3(cp, detail::sub3(v0, centroid));
    }

    return Polyhedron3D{std::move(out_vertices), std::move(out_facets), std::abs(total_vol),
                        total_area};
}

} // namespace algoat::geometry
