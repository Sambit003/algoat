/**
 * @file delaunay.hpp
 * @brief Modern, cache-aligned 2D Delaunay Triangulation and Voronoi Dual engine.
 *
 * Implements Bowyer-Watson incremental insertion guided by Morton Z-order presorting
 * and Jonathan Richard Shewchuk's exact adaptive geometric predicates (orient2d, incircle).
 *
 * Microarchitectural Highlights:
 * - Morton Z-Order presorting (via hardware BMI2 _pdep_u64) enables O(1) amortized
 *   jump-and-walk point location.
 * - Flat contiguous Struct-of-Arrays (SoA) layout eliminates pointer chasing and heap
 * fragmentation.
 * - Exact area-conserving Voronoi dual polygon generation via Sutherland-Hodgman clipping.
 */

#pragma once

#include "algoat/geometry/predicates.hpp"
#include "algoat/numerics/morton.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace algoat::geometry {

inline constexpr uint32_t INVALID_INDEX = 0xFFFFFFFFu;

/**
 * @brief Cache-aligned 2D Delaunay triangle representation.
 *
 * Stored in contiguous flat arrays to ensure zero heap fragmentation and optimal
 * L1/L2 cache locality during edge hops and cavity updates.
 */
struct alignas(32) Triangle {
    uint32_t a{INVALID_INDEX};      ///< Index into vertex array (CCW order)
    uint32_t b{INVALID_INDEX};      ///< Index into vertex array (CCW order)
    uint32_t c{INVALID_INDEX};      ///< Index into vertex array (CCW order)
    uint32_t ab_opp{INVALID_INDEX}; ///< Opposite triangle index sharing edge a->b
    uint32_t bc_opp{INVALID_INDEX}; ///< Opposite triangle index sharing edge b->c
    uint32_t ca_opp{INVALID_INDEX}; ///< Opposite triangle index sharing edge c->a
    Point2D circumcenter{0.0, 0.0}; ///< Triangle circumcircle center
    double circumradius{0.0};       ///< Triangle circumcircle radius
};

/**
 * @brief Voronoi cell dual polygon corresponding to a Delaunay site.
 */
struct VoronoiCell {
    uint32_t site_index{0};                ///< Site vertex index in input points
    std::vector<Point2D> polygon_vertices; ///< Ordered counterclockwise cell vertices
    bool is_bounded{true};                 ///< True if strictly bounded in the plane
};

namespace detail {

/**
 * @brief Squared Euclidean distance between two 2D points.
 */
[[nodiscard]] constexpr double squared_distance(const Point2D& p1, const Point2D& p2) noexcept {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

/**
 * @brief Translation-invariant circumcircle representation.
 */
struct Circumcircle {
    Point2D center{0.0, 0.0};
    double radius{0.0};
};

/**
 * @brief Translation-invariant circumcenter and circumradius computation.
 */
[[nodiscard]] inline Circumcircle compute_circumcircle(const Point2D& pa, const Point2D& pb,
                                                       const Point2D& pc) noexcept {
    const double ux = pb.x - pa.x;
    const double uy = pb.y - pa.y;
    const double vx = pc.x - pa.x;
    const double vy = pc.y - pa.y;

    const double du = ux * ux + uy * uy;
    const double dv = vx * vx + vy * vy;
    const double det = 2.0 * (ux * vy - uy * vx);

    if (std::abs(det) < 1e-15) {
        return Circumcircle{{(pa.x + pb.x + pc.x) / 3.0, (pa.y + pb.y + pc.y) / 3.0}, 0.0};
    }

    Point2D center{pa.x + (vy * du - uy * dv) / det, pa.y + (ux * dv - vx * du) / det};
    const double dx = center.x - pa.x;
    const double dy = center.y - pa.y;
    return Circumcircle{center, std::sqrt(dx * dx + dy * dy)};
}

/**
 * @brief Sutherland-Hodgman polygon clipping against a linear halfplane: A*x + B*y + C >= 0.
 */
inline void clip_polygon_halfplane(std::vector<Point2D>& poly, double a_coeff, double b_coeff,
                                   double c_coeff) {
    if (poly.empty()) {
        return;
    }

    std::vector<Point2D> output;
    output.reserve(poly.size() + 2);

    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D& s = poly[i];
        const Point2D& e = poly[(i + 1) % n];

        const double ds = a_coeff * s.x + b_coeff * s.y + c_coeff;
        const double de = a_coeff * e.x + b_coeff * e.y + c_coeff;

        constexpr double eps = 1e-13;
        const bool s_in = (ds >= -eps);
        const bool e_in = (de >= -eps);

        if (e_in) {
            if (!s_in) {
                const double t = ds / (ds - de);
                output.push_back({s.x + t * (e.x - s.x), s.y + t * (e.y - s.y)});
            }
            output.push_back(e);
        } else if (s_in) {
            const double t = ds / (ds - de);
            output.push_back({s.x + t * (e.x - s.x), s.y + t * (e.y - s.y)});
        }
    }

    poly = std::move(output);
}

} // namespace detail

/**
 * @class DelaunayTriangulation2D
 * @brief High-performance 2D Delaunay Triangulation and Voronoi Dual engine.
 */
class DelaunayTriangulation2D {
public:
    DelaunayTriangulation2D() = default;

    /**
     * @brief Constructs Delaunay triangulation from 2D points in O(N log N) time.
     *
     * @param points Contiguous span of 2D input vertices.
     */
    void build(std::span<const Point2D> points) {
        clear();
        if (points.empty()) {
            return;
        }

        vertices_.assign(points.begin(), points.end());
        const std::size_t n = points.size();
        if (n < 3) {
            return;
        }

        // 1. Compute Axis-Aligned Bounding Box
        double min_x = points[0].x;
        double max_x = points[0].x;
        double min_y = points[0].y;
        double max_y = points[0].y;

        for (std::size_t i = 1; i < n; ++i) {
            min_x = std::min(min_x, points[i].x);
            max_x = std::max(max_x, points[i].x);
            min_y = std::min(min_y, points[i].y);
            max_y = std::max(max_y, points[i].y);
        }

        const double dx = max_x - min_x;
        const double dy = max_y - min_y;
        const double d_max = std::max(dx, dy);

        if (d_max < 1e-14) {
            return; // All points are identical
        }

        // Check collinearity
        std::size_t p0_idx = 0;
        std::size_t p1_idx = INVALID_INDEX;
        for (std::size_t i = 1; i < n; ++i) {
            const double dist_sq =
                (points[i].x - points[p0_idx].x) * (points[i].x - points[p0_idx].x) +
                (points[i].y - points[p0_idx].y) * (points[i].y - points[p0_idx].y);
            if (dist_sq > 1e-24) {
                p1_idx = i;
                break;
            }
        }

        if (p1_idx == INVALID_INDEX) {
            return;
        }

        bool has_non_collinear = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (orient2d(points[p0_idx], points[p1_idx], points[i]) != Orientation::Collinear) {
                has_non_collinear = true;
                break;
            }
        }
        if (!has_non_collinear) {
            return; // All points are collinear; no 2D triangulation exists
        }

        // 2. Morton Z-Order Point Presorting
        struct PointKey {
            uint64_t key;
            uint32_t idx;
        };

        std::vector<PointKey> keys(n);
        for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i) {
            double nx = (points[i].x - min_x) / d_max;
            double ny = (points[i].y - min_y) / d_max;
            nx = std::clamp(nx, 0.0, 1.0);
            ny = std::clamp(ny, 0.0, 1.0);
            const auto ux = static_cast<uint32_t>(nx * 2147483647.0);
            const auto uy = static_cast<uint32_t>(ny * 2147483647.0);
            keys[i] = {algoat::numerics::morton_interleave(ux, uy), i};
        }

        if (keys.size() < 256) {
            std::sort(keys.begin(), keys.end(),
                      [](const PointKey& l, const PointKey& r) noexcept { return l.key < r.key; });
        } else {
            // 4-pass 16-bit Radix Sort across 64-bit Morton keys
            std::vector<PointKey> temp(keys.size());
            PointKey* src = keys.data();
            PointKey* dst = temp.data();
            const std::size_t total_keys = keys.size();
            std::vector<std::size_t> count(65536);

            for (int shift = 0; shift < 64; shift += 16) {
                std::fill(count.begin(), count.end(), 0);
                for (std::size_t i = 0; i < total_keys; ++i) {
                    count[(src[i].key >> shift) & 0xFFFF]++;
                }
                std::size_t total = 0;
                for (int i = 0; i < 65536; ++i) {
                    const std::size_t c = count[i];
                    count[i] = total;
                    total += c;
                }
                for (std::size_t i = 0; i < total_keys; ++i) {
                    const std::size_t bucket = (src[i].key >> shift) & 0xFFFF;
                    dst[count[bucket]++] = src[i];
                }
                std::swap(src, dst);
            }
            if (src != keys.data()) {
                std::copy(temp.begin(), temp.end(), keys.begin());
            }
        }

        // 3. Super-Triangle Initialization
        std::vector<Point2D> pts = vertices_;
        const double x_mid = 0.5 * (min_x + max_x);
        const double y_mid = 0.5 * (min_y + max_y);
        constexpr double M = 50.0;

        Point2D s0{x_mid - M * d_max, y_mid - d_max};
        Point2D s1{x_mid + M * d_max, y_mid - d_max};
        Point2D s2{x_mid, y_mid + M * d_max};

        if (orient2d(s0, s1, s2) != Orientation::CounterClockwise) {
            std::swap(s1, s2);
        }

        const auto s0_idx = static_cast<uint32_t>(pts.size());
        const auto s1_idx = static_cast<uint32_t>(pts.size() + 1);
        const auto s2_idx = static_cast<uint32_t>(pts.size() + 2);
        pts.push_back(s0);
        pts.push_back(s1);
        pts.push_back(s2);

        // 4. Flat Compact Triangle Mesh
        struct InternalTri {
            uint32_t v[3];   ///< v0, v1, v2 (CCW)
            uint32_t opp[3]; ///< opp[0]: opposite v0 (v1->v2), opp[1]: opposite v1 (v2->v0),
                             ///< opp[2]: opposite v2 (v0->v1)
            bool active{true};
        };

        std::vector<InternalTri> tri_pool;
        tri_pool.reserve(2 * n + 32);
        tri_pool.push_back(InternalTri{
            {s0_idx, s1_idx, s2_idx}, {INVALID_INDEX, INVALID_INDEX, INVALID_INDEX}, true});

        uint32_t recent_tri = 0;

        // BFS cavity scratchpad buffers
        std::vector<uint32_t> cavity;
        cavity.reserve(32);
        std::vector<uint32_t> cavity_queue;
        cavity_queue.reserve(32);
        struct BoundaryEdge {
            uint32_t u;
            uint32_t v;
            uint32_t opp_tri;
            uint32_t opp_edge;
        };
        std::vector<BoundaryEdge> boundary;
        boundary.reserve(32);
        std::vector<uint32_t> new_tri_indices;
        new_tri_indices.reserve(32);

        std::vector<uint32_t> tri_stamp(2 * n + 128, 0);
        uint32_t current_stamp = 0;

        // 5. Incremental Bowyer-Watson Insertion
        for (const auto& k : keys) {
            const uint32_t p_idx = k.idx;
            const Point2D& p = pts[p_idx];

            // A. Jump-and-Walk Point Location
            uint32_t curr = recent_tri;
            if (curr >= tri_pool.size() || !tri_pool[curr].active) {
                curr = 0;
            }

            uint32_t steps = 0;
            const uint32_t max_steps = static_cast<uint32_t>(tri_pool.size()) + 200;

            while (curr != INVALID_INDEX && steps++ < max_steps) {
                const auto& t = tri_pool[curr];

                const auto o0 = orient2d(pts[t.v[1]], pts[t.v[2]], p);
                if (o0 == Orientation::Clockwise && t.opp[0] != INVALID_INDEX) {
                    curr = t.opp[0];
                    continue;
                }

                const auto o1 = orient2d(pts[t.v[2]], pts[t.v[0]], p);
                if (o1 == Orientation::Clockwise && t.opp[1] != INVALID_INDEX) {
                    curr = t.opp[1];
                    continue;
                }

                const auto o2 = orient2d(pts[t.v[0]], pts[t.v[1]], p);
                if (o2 == Orientation::Clockwise && t.opp[2] != INVALID_INDEX) {
                    curr = t.opp[2];
                    continue;
                }

                break; // Located inside current triangle (or reached boundary)
            }

            if (curr == INVALID_INDEX || !tri_pool[curr].active) {
                // Exhaustive scan fallback
                for (uint32_t ti = 0; ti < static_cast<uint32_t>(tri_pool.size()); ++ti) {
                    if (!tri_pool[ti].active) {
                        continue;
                    }
                    const auto& t = tri_pool[ti];
                    if (orient2d(pts[t.v[1]], pts[t.v[2]], p) != Orientation::Clockwise &&
                        orient2d(pts[t.v[2]], pts[t.v[0]], p) != Orientation::Clockwise &&
                        orient2d(pts[t.v[0]], pts[t.v[1]], p) != Orientation::Clockwise) {
                        curr = ti;
                        break;
                    }
                }
                if (curr == INVALID_INDEX) {
                    continue;
                }
            }

            // Check if p is identical to any vertex of the containing triangle
            const auto& ct = tri_pool[curr];
            bool duplicate = false;
            for (int i = 0; i < 3; ++i) {
                const auto& pv = pts[ct.v[i]];
                if (p.x == pv.x && p.y == pv.y) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            // B. BFS Cavity Search
            if (tri_stamp.size() <= tri_pool.size() + 32) {
                tri_stamp.resize(tri_pool.size() + 1024, 0);
            }
            ++current_stamp;

            cavity.clear();
            boundary.clear();
            cavity_queue.clear();

            cavity_queue.push_back(curr);
            tri_stamp[curr] = current_stamp;

            std::size_t head = 0;
            while (head < cavity_queue.size()) {
                const uint32_t t_idx = cavity_queue[head++];
                cavity.push_back(t_idx);
                const auto& t = tri_pool[t_idx];

                for (uint32_t e = 0; e < 3; ++e) {
                    const uint32_t u = t.v[(e + 1) % 3];
                    const uint32_t v = t.v[(e + 2) % 3];
                    const uint32_t nbr = t.opp[e];

                    if (nbr != INVALID_INDEX) {
                        if (tri_stamp[nbr] == current_stamp) {
                            continue;
                        }
                        const auto& nt = tri_pool[nbr];
                        if (incircle(pts[nt.v[0]], pts[nt.v[1]], pts[nt.v[2]], p) ==
                            CircleLocation::Inside) {
                            tri_stamp[nbr] = current_stamp;
                            cavity_queue.push_back(nbr);
                        } else {
                            uint32_t nbr_e = INVALID_INDEX;
                            for (uint32_t ne = 0; ne < 3; ++ne) {
                                if (nt.opp[ne] == t_idx) {
                                    nbr_e = ne;
                                    break;
                                }
                            }
                            boundary.push_back({u, v, nbr, nbr_e});
                        }
                    } else {
                        boundary.push_back({u, v, INVALID_INDEX, INVALID_INDEX});
                    }
                }
            }

            // C. Cavity Retriangulation
            const std::size_t num_new = boundary.size();
            new_tri_indices.resize(num_new);

            for (std::size_t i = 0; i < num_new; ++i) {
                if (i < cavity.size()) {
                    new_tri_indices[i] = cavity[i];
                    tri_pool[cavity[i]].active = true;
                } else {
                    new_tri_indices[i] = static_cast<uint32_t>(tri_pool.size());
                    tri_pool.push_back(InternalTri{
                        {0, 0, 0}, {INVALID_INDEX, INVALID_INDEX, INVALID_INDEX}, true});
                }
            }

            // Deactivate leftover cavity triangles
            for (std::size_t i = num_new; i < cavity.size(); ++i) {
                tri_pool[cavity[i]].active = false;
            }

            // Populate newly created triangles
            for (std::size_t i = 0; i < num_new; ++i) {
                const auto& b_edge = boundary[i];
                auto& nt = tri_pool[new_tri_indices[i]];
                nt.v[0] = b_edge.u;
                nt.v[1] = b_edge.v;
                nt.v[2] = p_idx;
                nt.opp[2] = b_edge.opp_tri;

                if (b_edge.opp_tri != INVALID_INDEX && b_edge.opp_edge != INVALID_INDEX) {
                    tri_pool[b_edge.opp_tri].opp[b_edge.opp_edge] = new_tri_indices[i];
                }
            }

            // Connect radial edges between adjacent new triangles
            for (std::size_t i = 0; i < num_new; ++i) {
                const uint32_t vi = boundary[i].v;
                for (std::size_t j = 0; j < num_new; ++j) {
                    if (boundary[j].u == vi) {
                        tri_pool[new_tri_indices[i]].opp[0] = new_tri_indices[j];
                        tri_pool[new_tri_indices[j]].opp[1] = new_tri_indices[i];
                        break;
                    }
                }
            }

            recent_tri = new_tri_indices[0];
        }

        // 6. Remove Super-Triangle Vertices and Prune Adjacencies
        for (std::size_t i = 0; i < tri_pool.size(); ++i) {
            auto& t = tri_pool[i];
            if (!t.active) {
                continue;
            }
            if (t.v[0] >= n || t.v[1] >= n || t.v[2] >= n) {
                t.active = false;
                for (int e = 0; e < 3; ++e) {
                    if (t.opp[e] != INVALID_INDEX) {
                        auto& nbr = tri_pool[t.opp[e]];
                        for (int ne = 0; ne < 3; ++ne) {
                            if (nbr.opp[ne] == static_cast<uint32_t>(i)) {
                                nbr.opp[ne] = INVALID_INDEX;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // 7. Compact Surviving Triangles into Output
        std::vector<uint32_t> old_to_new(tri_pool.size(), INVALID_INDEX);
        uint32_t active_count = 0;
        for (std::size_t i = 0; i < tri_pool.size(); ++i) {
            if (tri_pool[i].active) {
                old_to_new[i] = active_count++;
            }
        }

        triangles_.reserve(active_count);
        for (std::size_t i = 0; i < tri_pool.size(); ++i) {
            const auto& t = tri_pool[i];
            if (!t.active) {
                continue;
            }

            Triangle final_t;
            final_t.a = t.v[0];
            final_t.b = t.v[1];
            final_t.c = t.v[2];
            final_t.bc_opp = (t.opp[0] != INVALID_INDEX) ? old_to_new[t.opp[0]] : INVALID_INDEX;
            final_t.ca_opp = (t.opp[1] != INVALID_INDEX) ? old_to_new[t.opp[1]] : INVALID_INDEX;
            final_t.ab_opp = (t.opp[2] != INVALID_INDEX) ? old_to_new[t.opp[2]] : INVALID_INDEX;

            const auto circ = detail::compute_circumcircle(
                vertices_[final_t.a], vertices_[final_t.b], vertices_[final_t.c]);
            final_t.circumcenter = circ.center;
            final_t.circumradius = circ.radius;
            triangles_.push_back(final_t);
        }

        super_triangle_root_ =
            triangles_.empty() ? 0u : static_cast<uint32_t>(triangles_.size() / 2);
    }

    /**
     * @brief Extracts read-only view of Delaunay triangles.
     */
    [[nodiscard]] std::span<const Triangle> triangles() const noexcept {
        return triangles_;
    }

    /**
     * @brief Generates the Voronoi Diagram dual with exact area conservation.
     */
    [[nodiscard]] std::vector<VoronoiCell> compute_voronoi_cells() const {
        const std::size_t n = vertices_.size();
        if (n == 0) {
            return {};
        }

        if (triangles_.empty() || n < 3) {
            std::vector<VoronoiCell> result(n);
            for (std::size_t i = 0; i < n; ++i) {
                result[i].site_index = static_cast<uint32_t>(i);
                result[i].is_bounded = false;
            }
            return result;
        }

        // Bounding box computation
        double min_x = vertices_[0].x;
        double max_x = vertices_[0].x;
        double min_y = vertices_[0].y;
        double max_y = vertices_[0].y;

        for (std::size_t i = 1; i < n; ++i) {
            min_x = std::min(min_x, vertices_[i].x);
            max_x = std::max(max_x, vertices_[i].x);
            min_y = std::min(min_y, vertices_[i].y);
            max_y = std::max(max_y, vertices_[i].y);
        }

        if (min_x == max_x) {
            max_x += 1.0;
        }
        if (min_y == max_y) {
            max_y += 1.0;
        }

        // Extract adjacency list & identify convex hull vertices
        std::vector<std::vector<uint32_t>> adj(n);
        std::vector<bool> is_hull(n, false);

        for (const auto& t : triangles_) {
            adj[t.a].push_back(t.b);
            adj[t.a].push_back(t.c);
            adj[t.b].push_back(t.a);
            adj[t.b].push_back(t.c);
            adj[t.c].push_back(t.a);
            adj[t.c].push_back(t.b);

            if (t.ab_opp == INVALID_INDEX) {
                is_hull[t.a] = true;
                is_hull[t.b] = true;
            }
            if (t.bc_opp == INVALID_INDEX) {
                is_hull[t.b] = true;
                is_hull[t.c] = true;
            }
            if (t.ca_opp == INVALID_INDEX) {
                is_hull[t.c] = true;
                is_hull[t.a] = true;
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            auto& neighbors = adj[i];
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }

        std::vector<VoronoiCell> cells(n);

        for (std::size_t i = 0; i < n; ++i) {
            cells[i].site_index = static_cast<uint32_t>(i);

            if (adj[i].empty()) {
                cells[i].polygon_vertices.clear();
                cells[i].is_bounded = false;
                continue;
            }

            cells[i].is_bounded = !is_hull[i];

            // Start with bounding box in CCW order
            std::vector<Point2D> poly = {
                {min_x, min_y}, {max_x, min_y}, {max_x, max_y}, {min_x, max_y}};

            const Point2D& pi = vertices_[i];
            for (const uint32_t j : adj[i]) {
                const Point2D& pj = vertices_[j];
                const double vx = pj.x - pi.x;
                const double vy = pj.y - pi.y;
                const double mx = 0.5 * (pi.x + pj.x);
                const double my = 0.5 * (pi.y + pj.y);

                // Halfplane: -vx * x - vy * y + (vx * mx + vy * my) >= 0
                const double a_coeff = -vx;
                const double b_coeff = -vy;
                const double c_coeff = vx * mx + vy * my;

                detail::clip_polygon_halfplane(poly, a_coeff, b_coeff, c_coeff);
                if (poly.empty()) {
                    break;
                }
            }

            cells[i].polygon_vertices = std::move(poly);
        }

        return cells;
    }

    /**
     * @brief Point location: returns the index of the triangle containing the query point.
     *
     * @param query 2D coordinates of query point.
     * @return Optional triangle index, or std::nullopt if query lies outside the convex hull.
     */
    [[nodiscard]] std::optional<uint32_t> locate_point(const Point2D& query) const noexcept {
        if (triangles_.empty()) {
            return std::nullopt;
        }

        uint32_t curr = super_triangle_root_;
        if (curr >= triangles_.size()) {
            curr = 0;
        }

        uint32_t steps = 0;
        const uint32_t max_steps = static_cast<uint32_t>(triangles_.size()) + 200;

        while (curr != INVALID_INDEX && steps++ < max_steps) {
            const auto& t = triangles_[curr];
            const Point2D& pa = vertices_[t.a];
            const Point2D& pb = vertices_[t.b];
            const Point2D& pc = vertices_[t.c];

            const auto o_ab = orient2d(pa, pb, query);
            const auto o_bc = orient2d(pb, pc, query);
            const auto o_ca = orient2d(pc, pa, query);

            if (o_ab != Orientation::Clockwise && o_bc != Orientation::Clockwise &&
                o_ca != Orientation::Clockwise) {
                return curr;
            }

            if (o_ab == Orientation::Clockwise) {
                if (t.ab_opp == INVALID_INDEX) {
                    return std::nullopt; // Outside convex hull
                }
                curr = t.ab_opp;
            } else if (o_bc == Orientation::Clockwise) {
                if (t.bc_opp == INVALID_INDEX) {
                    return std::nullopt; // Outside convex hull
                }
                curr = t.bc_opp;
            } else if (o_ca == Orientation::Clockwise) {
                if (t.ca_opp == INVALID_INDEX) {
                    return std::nullopt; // Outside convex hull
                }
                curr = t.ca_opp;
            } else {
                break;
            }
        }

        return std::nullopt;
    }

    /**
     * @brief Nearest vertex query in O(log N) average time.
     *
     * @param query 2D coordinates of query point.
     * @return Index of nearest input vertex.
     */
    [[nodiscard]] uint32_t nearest_site(const Point2D& query) const noexcept {
        const std::size_t n = vertices_.size();
        if (n == 0) {
            return INVALID_INDEX;
        }
        if (n == 1) {
            return 0;
        }

        if (triangles_.empty() || n < 16) {
            return brute_force_nearest_site(query);
        }

        const auto opt_tri = locate_point(query);
        if (opt_tri.has_value()) {
            uint32_t curr_tri = *opt_tri;
            uint32_t best_idx = triangles_[curr_tri].a;
            double best_dist = detail::squared_distance(vertices_[best_idx], query);

            for (uint32_t v :
                 {triangles_[curr_tri].a, triangles_[curr_tri].b, triangles_[curr_tri].c}) {
                const double d = detail::squared_distance(vertices_[v], query);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx = v;
                }
            }

            // Hill-climb across adjacent triangles
            bool improved = true;
            uint32_t steps = 0;
            while (improved && steps++ < 32) {
                improved = false;
                const auto& t = triangles_[curr_tri];
                for (uint32_t nbr_idx : {t.ab_opp, t.bc_opp, t.ca_opp}) {
                    if (nbr_idx != INVALID_INDEX) {
                        const auto& nt = triangles_[nbr_idx];
                        for (uint32_t nv : {nt.a, nt.b, nt.c}) {
                            const double d = detail::squared_distance(vertices_[nv], query);
                            if (d < best_dist) {
                                best_dist = d;
                                best_idx = nv;
                                curr_tri = nbr_idx;
                                improved = true;
                                break;
                            }
                        }
                        if (improved) {
                            break;
                        }
                    }
                }
            }

            return best_idx;
        }

        // Query lies outside the convex hull: search among boundary/extrema
        return brute_force_nearest_site(query);
    }

    [[nodiscard]] std::size_t vertex_count() const noexcept {
        return vertices_.size();
    }

    [[nodiscard]] std::size_t triangle_count() const noexcept {
        return triangles_.size();
    }

    void clear() noexcept {
        vertices_.clear();
        triangles_.clear();
        super_triangle_root_ = 0;
    }

private:
    [[nodiscard]] uint32_t brute_force_nearest_site(const Point2D& query) const noexcept {
        const std::size_t n = vertices_.size();
        if (n == 0) {
            return INVALID_INDEX;
        }
        uint32_t best_idx = 0;
        double best_dist_sq = detail::squared_distance(vertices_[0], query);
        for (std::size_t i = 1; i < n; ++i) {
            const double d_sq = detail::squared_distance(vertices_[i], query);
            if (d_sq < best_dist_sq) {
                best_dist_sq = d_sq;
                best_idx = static_cast<uint32_t>(i);
            }
        }
        return best_idx;
    }

    std::vector<Point2D> vertices_;
    std::vector<Triangle> triangles_;
    uint32_t super_triangle_root_{0};
};

} // namespace algoat::geometry
