/**
 * @file polygon_boolean.hpp
 * @brief Robust 2D Polygon Boolean Operations via Martínez–Rueda–Feito Sweep-Line
 *        and Shewchuk Adaptive Floating-Point Predicates.
 *
 * Implements Boolean set operations (Intersection, Union, Difference, Xor) on
 * arbitrary 2D polygons including concave shapes, polygons with multiple nested
 * holes/islands, disjoint multi-polygons, self-intersecting rings, and collinear
 * overlapping edges in O((N + K) log N) time.
 *
 * Note on symbol disambiguation:
 * `algoat::geometry::Polygon` is already used by `algoat/geometry/polygon.hpp` for
 * the SIMD Point-in-Polygon classifier. To avoid C++ redeclaration conflicts when
 * both headers are included (e.g. via `algoat/algoat.hpp` or `import algoat;`),
 * this header exposes `BooleanPolygon<PointType>` in `algoat::geometry` alongside
 * `algoat::geometry::boolean::Polygon<PointType>` as an alias.
 */

#pragma once

#include "algoat/geometry/predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <set>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::geometry {

/**
 * @brief Concept for 2D point types supporting either `[0]`/`[1]` indexing or `.x`/`.y` fields.
 */
template <typename P>
concept Point2DLike = (requires(const P& p) {
    { p[0] } -> std::convertible_to<double>;
    { p[1] } -> std::convertible_to<double>;
} || requires(const P& p) {
    { p.x } -> std::convertible_to<double>;
    { p.y } -> std::convertible_to<double>;
});

/**
 * @brief Supported 2D polygon Boolean set operations.
 */
enum class BooleanOp : uint8_t { Intersection, Union, Difference, Xor };

namespace detail {

inline constexpr uint32_t kBoolInvalidIdx = 0xFFFFFFFFu;

template <Point2DLike PointType> [[nodiscard]] constexpr double get_x(const PointType& p) noexcept {
    if constexpr (requires { p.x; }) {
        return static_cast<double>(p.x);
    } else {
        return static_cast<double>(p[0]);
    }
}

template <Point2DLike PointType> [[nodiscard]] constexpr double get_y(const PointType& p) noexcept {
    if constexpr (requires { p.y; }) {
        return static_cast<double>(p.y);
    } else {
        return static_cast<double>(p[1]);
    }
}

template <Point2DLike PointType>
[[nodiscard]] constexpr PointType make_point(double x, double y) noexcept {
    if constexpr (std::is_constructible_v<PointType, double, double>) {
        return PointType{x, y};
    } else if constexpr (requires(PointType p) {
                             p.x = x;
                             p.y = y;
                         }) {
        PointType p{};
        p.x = x;
        p.y = y;
        return p;
    } else {
        PointType p{};
        p[0] = x;
        p[1] = y;
        return p;
    }
}

/**
 * @brief Computes translation-invariant signed area of a closed 2D ring via Kahan-compensated
 *        shoelace summation (positive for CCW, negative for CW).
 */
template <Point2DLike PointType>
[[nodiscard]] inline double ring_signed_area(std::span<const PointType> ring) noexcept {
    const std::size_t n = ring.size();
    if (n < 3) {
        return 0.0;
    }
    const double x0 = get_x(ring[0]);
    const double y0 = get_y(ring[0]);
    double sum = 0.0;
    double compensation = 0.0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double xi = get_x(ring[i]) - x0;
        const double yi = get_y(ring[i]) - y0;
        const double xi1 = get_x(ring[i + 1]) - x0;
        const double yi1 = get_y(ring[i + 1]) - y0;
        const double cross = std::fma(xi, yi1, -xi1 * yi);
        const double t = sum + cross;
        if (std::abs(sum) >= std::abs(cross)) {
            compensation += (sum - t) + cross;
        } else {
            compensation += (cross - t) + sum;
        }
        sum = t;
    }
    return 0.5 * (sum + compensation);
}

} // namespace detail

/**
 * @brief Polygon with an outer boundary ring and zero or more interior hole rings.
 *
 * @tparam PointType 2D point type satisfying @c Point2DLike (defaults to @c std::array<double, 2>).
 */
template <Point2DLike PointType = std::array<double, 2>> class BooleanPolygon {
public:
    using point_type = PointType;

    std::vector<PointType> outer_ring;
    std::vector<std::vector<PointType>> holes;

    /// @brief Returns true if the outer boundary ring is empty.
    [[nodiscard]] bool empty() const noexcept {
        return outer_ring.empty();
    }

    /// @brief Computes the net unsigned area of the outer boundary minus all holes.
    [[nodiscard]] double area() const noexcept {
        const double outer_area = std::abs(detail::ring_signed_area<PointType>(outer_ring));
        double holes_area = 0.0;
        for (const auto& hole : holes) {
            holes_area += std::abs(detail::ring_signed_area<PointType>(hole));
        }
        return std::max(0.0, outer_area - holes_area);
    }
};

/// @brief Collection of disjoint polygons (possibly with holes) produced by boolean operations.
template <Point2DLike PointType = std::array<double, 2>>
using MultiPolygon = std::vector<BooleanPolygon<PointType>>;

namespace detail {

inline constexpr double kNearlyEqualRelTol = 1e-15;
inline constexpr double kNearlyOnSegBoxTol = 1e-12;
inline constexpr double kNearlyOnSegCrossSqTol = 1e-24;
inline constexpr double kUlpSegTolFactor = 32.0;
inline constexpr std::size_t kMinSweepIterations = 100000;
inline constexpr std::size_t kSweepIterationSlack = 4096;

[[nodiscard]] inline bool points_equal(const Point2D& a, const Point2D& b) noexcept {
    return a.x == b.x && a.y == b.y;
}

[[nodiscard]] inline bool point_less(const Point2D& a, const Point2D& b) noexcept {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
}

[[nodiscard]] inline bool nearly_equal_double(double a, double b, int64_t max_ulps = 4) noexcept {
    if (a == b) {
        return true;
    }
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    // Check tiny absolute difference around zero as well
    if (std::abs(a - b) <= kNearlyEqualRelTol * std::max({1.0, std::abs(a), std::abs(b)})) {
        return true;
    }
    auto to_ordered = [](double v) noexcept -> uint64_t {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        constexpr uint64_t kSignBit = 1ULL << 63;
        if ((bits & kSignBit) != 0) {
            return kSignBit - (bits & ~kSignBit);
        }
        return kSignBit + bits;
    };
    const uint64_t oa = to_ordered(a);
    const uint64_t ob = to_ordered(b);
    const uint64_t diff = (oa >= ob) ? (oa - ob) : (ob - oa);
    return diff <= static_cast<uint64_t>(max_ulps);
}

[[nodiscard]] inline bool nearly_equal_points(const Point2D& a, const Point2D& b,
                                              int64_t max_ulps = 4) noexcept {
    return nearly_equal_double(a.x, b.x, max_ulps) && nearly_equal_double(a.y, b.y, max_ulps);
}

[[nodiscard]] inline bool point_in_seg_box(const Point2D& p, const Point2D& a,
                                           const Point2D& b) noexcept {
    const double min_x = std::min(a.x, b.x);
    const double max_x = std::max(a.x, b.x);
    const double min_y = std::min(a.y, b.y);
    const double max_y = std::max(a.y, b.y);
    return p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y;
}

[[nodiscard]] inline bool nearly_on_segment(const Point2D& p, const Point2D& a,
                                            const Point2D& b) noexcept {
    if (!point_in_seg_box(p, a, b)) {
        const double scale =
            std::max({1.0, std::abs(a.x), std::abs(a.y), std::abs(b.x), std::abs(b.y)});
        const double tol = kNearlyOnSegBoxTol * scale;
        if (p.x < std::min(a.x, b.x) - tol || p.x > std::max(a.x, b.x) + tol ||
            p.y < std::min(a.y, b.y) - tol || p.y > std::max(a.y, b.y) + tol) {
            return false;
        }
    }
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double len_sq = dx * dx + dy * dy;
    if (len_sq == 0.0) {
        return nearly_equal_points(p, a);
    }
    const double cross = std::abs((p.x - a.x) * dy - (p.y - a.y) * dx);
    return (cross * cross) <= (kNearlyOnSegCrossSqTol * len_sq * len_sq);
}

/**
 * @brief Checks if `p` lies strictly within `a..b`'s bounding box and within ~32 ULPs of [a, b],
 *        catching endpoints that sit on a prior split intersection whose endpoints rounded by < 1
 * ULP.
 */
[[nodiscard]] inline bool ulp_on_segment(const Point2D& p, const Point2D& a,
                                         const Point2D& b) noexcept {
    if (!point_in_seg_box(p, a, b)) {
        return false;
    }
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double len = std::hypot(dx, dy);
    if (len == 0.0) {
        return false;
    }
    const double scale =
        std::max({1.0, std::abs(a.x), std::abs(a.y), std::abs(b.x), std::abs(b.y)});
    const double tol = kUlpSegTolFactor * std::numeric_limits<double>::epsilon() * scale * len;
    const double cross = std::abs((p.x - a.x) * dy - (p.y - a.y) * dx);
    return cross <= tol;
}

[[nodiscard]] inline int
check_shared_endpoint_overlap(const Point2D& shared, const Point2D& other_a, const Point2D& other_b,
                              const Point2D& p0, const Point2D& p1, const Point2D& q0,
                              const Point2D& q1, Point2D& ip0, Point2D& ip1) noexcept {
    if (nearly_on_segment(other_b, p0, p1) || nearly_on_segment(other_a, q0, q1)) {
        ip0 = shared;
        ip1 = point_in_seg_box(other_b, p0, p1) ? other_b : other_a;
        if (!points_equal(ip0, ip1)) {
            if (point_less(ip1, ip0)) {
                std::swap(ip0, ip1);
            }
            return 2;
        }
    }
    ip0 = shared;
    return 1;
}

/**
 * @brief Computes exact/clamped intersection between segments [p0, p1] and [q0, q1].
 * @return 0 if disjoint, 1 if single point (ip0), 2 if collinear overlap [ip0, ip1].
 */
[[nodiscard]] inline int segment_intersection(const Point2D& p0, const Point2D& p1,
                                              const Point2D& q0, const Point2D& q1, Point2D& ip0,
                                              Point2D& ip1) noexcept {
    const double p_min_x = std::min(p0.x, p1.x);
    const double p_max_x = std::max(p0.x, p1.x);
    const double p_min_y = std::min(p0.y, p1.y);
    const double p_max_y = std::max(p0.y, p1.y);

    const double q_min_x = std::min(q0.x, q1.x);
    const double q_max_x = std::max(q0.x, q1.x);
    const double q_min_y = std::min(q0.y, q1.y);
    const double q_max_y = std::max(q0.y, q1.y);

    if (p_max_x < q_min_x || q_max_x < p_min_x || p_max_y < q_min_y || q_max_y < p_min_y) {
        return 0;
    }

    const Orientation op0 = orient2d(p0, p1, q0);
    const Orientation op1 = orient2d(p0, p1, q1);
    const Orientation oq0 = orient2d(q0, q1, p0);
    const Orientation oq1 = orient2d(q0, q1, p1);

    // 1. Exact collinear case
    if (op0 == Orientation::Collinear && op1 == Orientation::Collinear) {
        Point2D plo = point_less(p0, p1) ? p0 : p1;
        Point2D phi = point_less(p0, p1) ? p1 : p0;
        Point2D qlo = point_less(q0, q1) ? q0 : q1;
        Point2D qhi = point_less(q0, q1) ? q1 : q0;

        Point2D lo = point_less(plo, qlo) ? qlo : plo;
        Point2D hi = point_less(phi, qhi) ? phi : qhi;

        if (point_less(hi, lo)) {
            return 0;
        }
        if (points_equal(lo, hi)) {
            ip0 = lo;
            return 1;
        }
        ip0 = lo;
        ip1 = hi;
        return 2;
    }

    // 2. Exact shared endpoint (and nearly-collinear overlap)
    if (points_equal(p0, q0) || points_equal(p0, q1)) {
        const Point2D& other_b = points_equal(p0, q0) ? q1 : q0;
        return check_shared_endpoint_overlap(p0, p1, other_b, p0, p1, q0, q1, ip0, ip1);
    }
    if (points_equal(p1, q0) || points_equal(p1, q1)) {
        const Point2D& other_b = points_equal(p1, q0) ? q1 : q0;
        return check_shared_endpoint_overlap(p1, p0, other_b, p0, p1, q0, q1, ip0, ip1);
    }

    if (op0 == Orientation::Collinear && point_in_seg_box(q0, p0, p1)) {
        ip0 = q0;
        return 1;
    }
    if (op1 == Orientation::Collinear && point_in_seg_box(q1, p0, p1)) {
        ip0 = q1;
        return 1;
    }
    if (oq0 == Orientation::Collinear && point_in_seg_box(p0, q0, q1)) {
        ip0 = p0;
        return 1;
    }
    if (oq1 == Orientation::Collinear && point_in_seg_box(p1, q0, q1)) {
        ip0 = p1;
        return 1;
    }

    // 3. ULP-near-collinear T-junction guard (recovers vertex incidence across 1-ULP split drift)
    if (ulp_on_segment(q0, p0, p1)) {
        ip0 = q0;
        return 1;
    }
    if (ulp_on_segment(q1, p0, p1)) {
        ip0 = q1;
        return 1;
    }
    if (ulp_on_segment(p0, q0, q1)) {
        ip0 = p0;
        return 1;
    }
    if (ulp_on_segment(p1, q0, q1)) {
        ip0 = p1;
        return 1;
    }

    // 4. Strict same-side rejection via exact Shewchuk orient2d
    if (op0 == op1 && op0 != Orientation::Collinear) {
        return 0;
    }
    if (oq0 == oq1 && oq0 != Orientation::Collinear) {
        return 0;
    }

    // 5. Proper interior crossing (local translation at p0)
    const double dx0 = p1.x - p0.x;
    const double dy0 = p1.y - p0.y;
    const double dx1 = q1.x - q0.x;
    const double dy1 = q1.y - q0.y;
    const double kross = dx0 * dy1 - dy0 * dx1;
    if (kross == 0.0) {
        return 0;
    }
    const double ex = q0.x - p0.x;
    const double ey = q0.y - p0.y;
    const double t = std::clamp((ex * dy1 - ey * dx1) / kross, 0.0, 1.0);

    Point2D cand{p0.x + t * dx0, p0.y + t * dy0};
    cand.x = std::clamp(cand.x, std::max(p_min_x, q_min_x), std::min(p_max_x, q_max_x));
    cand.y = std::clamp(cand.y, std::max(p_min_y, q_min_y), std::min(p_max_y, q_max_y));

    if (nearly_equal_points(cand, p0)) {
        cand = p0;
    } else if (nearly_equal_points(cand, p1)) {
        cand = p1;
    } else if (nearly_equal_points(cand, q0)) {
        cand = q0;
    } else if (nearly_equal_points(cand, q1)) {
        cand = q1;
    }

    ip0 = cand;
    return 1;
}

/**
 * @brief Sanitizes a polygon ring: removes consecutive duplicates and 180-degree spikes.
 */
template <Point2DLike PointType>
[[nodiscard]] inline std::vector<Point2D> sanitize_ring(std::span<const PointType> ring, double cx,
                                                        double cy) {
    std::vector<Point2D> pts;
    if (ring.size() < 3) {
        return pts;
    }
    pts.reserve(ring.size());
    for (const auto& p : ring) {
        const double raw_x = get_x(p);
        const double raw_y = get_y(p);
        if (!std::isfinite(raw_x) || !std::isfinite(raw_y)) {
            return {};
        }
        const Point2D pt{raw_x - cx, raw_y - cy};
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {
            return {};
        }
        if (pts.empty() || !points_equal(pts.back(), pt)) {
            pts.push_back(pt);
        }
    }
    while (pts.size() >= 2 && points_equal(pts.front(), pts.back())) {
        pts.pop_back();
    }
    // Cyclically remove consecutive duplicates and A -> B -> A spikes
    bool changed = true;
    while (changed && pts.size() >= 3) {
        changed = false;
        std::vector<Point2D> next;
        next.reserve(pts.size());
        const std::size_t n = pts.size();
        for (std::size_t i = 0; i < n; ++i) {
            const Point2D& prev = next.empty() ? pts[(i + n - 1) % n] : next.back();
            const Point2D& curr = pts[i];
            const Point2D& nxt = pts[(i + 1) % n];
            if (points_equal(curr, prev) || points_equal(curr, nxt) || points_equal(prev, nxt)) {
                changed = true;
                continue;
            }
            next.push_back(curr);
        }
        pts.swap(next);
    }
    if (pts.size() < 3) {
        pts.clear();
    }
    return pts;
}

enum class PolyId : uint8_t { Subject = 0, Clipping = 1 };

struct SweepEvent {
    Point2D point{};
    uint32_t other_event{kBoolInvalidIdx};
    PolyId pol{PolyId::Subject};
    bool left{false};
    bool dead{false};
    bool in_sl{false};
    uint8_t subj_parity{0};
    uint8_t clip_parity{0};
    bool below_subj{false};
    bool below_clip{false};
    bool above_subj{false};
    bool above_clip{false};
    bool in_result{false};
};

struct SweepEventComp {
    const std::vector<SweepEvent>* events{nullptr};

    // Priority queue max-heap comparator: returns true when `a` is processed AFTER `b`.
    [[nodiscard]] bool operator()(uint32_t a_idx, uint32_t b_idx) const noexcept {
        if (a_idx == b_idx) {
            return false;
        }
        const auto& a = (*events)[a_idx];
        const auto& b = (*events)[b_idx];
        if (a.point.x != b.point.x) {
            return a.point.x > b.point.x;
        }
        if (a.point.y != b.point.y) {
            return a.point.y > b.point.y;
        }
        if (a.left != b.left) {
            return a.left; // right (!left) processed before left
        }
        const auto& oa = (*events)[a.other_event].point;
        const auto& ob = (*events)[b.other_event].point;
        const Orientation o = orient2d(a.point, oa, ob);
        if (o != Orientation::Collinear) {
            return o == Orientation::Clockwise;
        }
        if (a.pol != b.pol) {
            return static_cast<uint8_t>(a.pol) > static_cast<uint8_t>(b.pol);
        }
        return a_idx > b_idx;
    }
};

struct SegmentComp {
    const std::vector<SweepEvent>* events{nullptr};

    // Status tree comparator: returns true iff segment `a` is strictly BELOW segment `b`.
    [[nodiscard]] bool operator()(uint32_t a_idx, uint32_t b_idx) const noexcept {
        if (a_idx == b_idx) {
            return false;
        }
        const auto& a = (*events)[a_idx];
        const auto& b = (*events)[b_idx];
        const Point2D& al = a.point;
        const Point2D& ar = (*events)[a.other_event].point;
        const Point2D& bl = b.point;
        const Point2D& br = (*events)[b.other_event].point;

        const Orientation o1 = orient2d(al, ar, bl);
        const Orientation o2 = orient2d(al, ar, br);

        if (o1 == Orientation::Collinear && o2 == Orientation::Collinear) {
            if (al.x != bl.x) {
                return al.x < bl.x;
            }
            if (al.y != bl.y) {
                return al.y < bl.y;
            }
            if (ar.x != br.x) {
                return ar.x < br.x;
            }
            if (ar.y != br.y) {
                return ar.y < br.y;
            }
            if (a.pol != b.pol) {
                return static_cast<uint8_t>(a.pol) < static_cast<uint8_t>(b.pol);
            }
            return a_idx < b_idx;
        }

        if (points_equal(al, bl)) {
            return o2 == Orientation::CounterClockwise;
        }
        if (al.x == bl.x) {
            return al.y < bl.y;
        }
        if (al.x < bl.x) {
            if (o1 != Orientation::Collinear) {
                return o1 == Orientation::CounterClockwise;
            }
            return o2 == Orientation::CounterClockwise;
        }
        const Orientation ob1 = orient2d(bl, br, al);
        if (ob1 != Orientation::Collinear) {
            return ob1 == Orientation::Clockwise;
        }
        const Orientation ob2 = orient2d(bl, br, ar);
        return ob2 == Orientation::Clockwise;
    }
};

class SweepEngine {
public:
    using StatusSet = std::set<uint32_t, SegmentComp>;

    explicit SweepEngine(BooleanOp op)
        : op_(op), eq_(SweepEventComp{&events_}), sl_(SegmentComp{&events_}) {}

    void add_ring(std::span<const Point2D> ring, PolyId pol) {
        const std::size_t n = ring.size();
        if (n < 3) {
            return;
        }
        for (std::size_t i = 0; i < n; ++i) {
            add_segment(ring[i], ring[(i + 1 == n) ? 0 : (i + 1)], pol);
        }
    }

    template <Point2DLike PointType>
    [[nodiscard]] MultiPolygon<PointType> run(double cx, double cy) {
        if (events_.empty()) {
            return {};
        }

        const std::size_t max_iterations = std::max<std::size_t>(
            kMinSweepIterations, events_.size() * events_.size() + kSweepIterationSlack);
        std::size_t iters = 0;

        while (!eq_.empty() && iters < max_iterations) {
            ++iters;
            const uint32_t e_idx = eq_.top();
            eq_.pop();

            if (events_[e_idx].dead) {
                continue;
            }

            if (events_[e_idx].left) {
                auto [it, inserted] = sl_.insert(e_idx);
                if (!inserted) {
                    continue;
                }
                events_[e_idx].in_sl = true;
                ensure_sl_capacity(e_idx);
                sl_its_[e_idx] = it;

                auto prev_it = (it != sl_.begin()) ? std::prev(it) : sl_.end();
                auto next_it = std::next(it);

                compute_fields(e_idx, prev_it);

                bool requeue = false;
                if (next_it != sl_.end()) {
                    const uint32_t next_idx = *next_it;
                    requeue = handle_intersection(e_idx, next_idx, e_idx);
                }
                if (!requeue && !events_[e_idx].dead && events_[e_idx].in_sl) {
                    auto cur_it = sl_its_[e_idx];
                    if (cur_it != sl_.begin()) {
                        const uint32_t prev_idx = *std::prev(cur_it);
                        requeue = handle_intersection(prev_idx, e_idx, e_idx);
                    }
                }
                if (requeue && !events_[e_idx].dead && events_[e_idx].in_sl) {
                    sl_.erase(sl_its_[e_idx]);
                    events_[e_idx].in_sl = false;
                    events_[e_idx].in_result = false;
                    eq_.push(e_idx);
                }
            } else {
                const uint32_t left_idx = events_[e_idx].other_event;
                if (events_[left_idx].dead || !events_[left_idx].in_sl) {
                    continue;
                }
                auto it = sl_its_[left_idx];
                auto prev_it = (it != sl_.begin()) ? std::prev(it) : sl_.end();
                auto next_it = std::next(it);

                if (events_[left_idx].in_result) {
                    result_segments_.push_back(left_idx);
                }

                sl_.erase(it);
                events_[left_idx].in_sl = false;

                if (prev_it != sl_.end() && next_it != sl_.end()) {
                    (void)handle_intersection(*prev_it, *next_it, kBoolInvalidIdx);
                }
            }
        }

        return reconstruct_multipolygon<PointType>(cx, cy);
    }

private:
    BooleanOp op_;
    std::vector<SweepEvent> events_;
    std::vector<StatusSet::iterator> sl_its_;
    std::priority_queue<uint32_t, std::vector<uint32_t>, SweepEventComp> eq_;
    StatusSet sl_;
    std::vector<uint32_t> result_segments_;

    void ensure_sl_capacity(uint32_t idx) {
        if (idx >= sl_its_.size()) {
            sl_its_.resize(events_.size());
        }
    }

    void add_segment(const Point2D& p1, const Point2D& p2, PolyId pol) {
        if (points_equal(p1, p2)) {
            return;
        }
        const uint32_t i1 = static_cast<uint32_t>(events_.size());
        const uint32_t i2 = i1 + 1;
        const bool p1_left = point_less(p1, p2);

        SweepEvent e1{};
        e1.point = p1;
        e1.other_event = i2;
        e1.pol = pol;
        e1.left = p1_left;
        e1.subj_parity = (pol == PolyId::Subject) ? 1 : 0;
        e1.clip_parity = (pol == PolyId::Clipping) ? 1 : 0;

        SweepEvent e2{};
        e2.point = p2;
        e2.other_event = i1;
        e2.pol = pol;
        e2.left = !p1_left;
        e2.subj_parity = (pol == PolyId::Subject) ? 1 : 0;
        e2.clip_parity = (pol == PolyId::Clipping) ? 1 : 0;

        events_.push_back(e1);
        events_.push_back(e2);
        ensure_sl_capacity(i2);
        eq_.push(i1);
        eq_.push(i2);
    }

    [[nodiscard]] bool is_vertical(uint32_t left_idx) const noexcept {
        const auto& l = events_[left_idx];
        const auto& r = events_[l.other_event];
        return l.point.x == r.point.x;
    }

    [[nodiscard]] bool in_set(bool subj_in, bool clip_in) const noexcept {
        switch (op_) {
        case BooleanOp::Intersection:
            return subj_in && clip_in;
        case BooleanOp::Union:
            return subj_in || clip_in;
        case BooleanOp::Difference:
            return subj_in && !clip_in;
        case BooleanOp::Xor:
            return subj_in != clip_in;
        }
        return false;
    }

    void compute_fields(uint32_t le_idx, StatusSet::iterator prev_it) noexcept {
        auto& le = events_[le_idx];
        if (prev_it == sl_.end()) {
            le.below_subj = false;
            le.below_clip = false;
        } else {
            const auto& prev = events_[*prev_it];
            le.below_subj = prev.above_subj;
            le.below_clip = prev.above_clip;
        }

        if (is_vertical(le_idx)) {
            // Vertical segments have zero horizontal span in (x, x + 0+), so they do not flip
            // state along x + 0+; their own left vs right transition flips across x = x0.
            le.above_subj = le.below_subj;
            le.above_clip = le.below_clip;
            const bool right_in = in_set(le.below_subj, le.below_clip);
            const bool left_subj = le.below_subj ^ (le.subj_parity != 0);
            const bool left_clip = le.below_clip ^ (le.clip_parity != 0);
            const bool left_in = in_set(left_subj, left_clip);
            le.in_result = (left_in != right_in);
        } else {
            le.above_subj = le.below_subj ^ (le.subj_parity != 0);
            le.above_clip = le.below_clip ^ (le.clip_parity != 0);
            const bool below_in = in_set(le.below_subj, le.below_clip);
            const bool above_in = in_set(le.above_subj, le.above_clip);
            le.in_result = (below_in != above_in);
        }
    }

    void recompute_fields_in_sl(uint32_t le_idx) noexcept {
        if (events_[le_idx].dead || !events_[le_idx].in_sl) {
            return;
        }
        auto it = sl_its_[le_idx];
        auto prev_it = (it != sl_.begin()) ? std::prev(it) : sl_.end();
        compute_fields(le_idx, prev_it);
    }

    /**
     * @brief Splits left event `le_idx` at `ip` safely (removing from `sl_` before mutation).
     * @return true if a newly created event at `ip` sorts before `ref_idx` at the same point.
     */
    bool divide_segment(uint32_t le_idx, Point2D ip, uint32_t ref_idx) {
        const Point2D l_pt = events_[le_idx].point;
        const uint32_t old_r_idx = events_[le_idx].other_event;
        const Point2D r_pt = events_[old_r_idx].point;

        // Clamp ip within [l_pt, r_pt] sweep bounds
        if (ip.x < l_pt.x) {
            ip.x = l_pt.x;
        } else if (ip.x > r_pt.x) {
            ip.x = r_pt.x;
        }
        if (ip.x == l_pt.x && ip.y < l_pt.y) {
            if (l_pt.x < r_pt.x) {
                ip.x = std::nextafter(l_pt.x, r_pt.x);
            } else {
                ip.y = l_pt.y;
            }
        }
        if (ip.x == r_pt.x && ip.y > r_pt.y) {
            if (l_pt.x < r_pt.x) {
                ip.x = std::nextafter(r_pt.x, l_pt.x);
            } else {
                ip.y = r_pt.y;
            }
        }

        if (points_equal(ip, l_pt) || points_equal(ip, r_pt)) {
            return false;
        }

        const bool was_in_sl = events_[le_idx].in_sl;
        if (was_in_sl) {
            sl_.erase(sl_its_[le_idx]);
            events_[le_idx].in_sl = false;
        }

        const uint32_t r_new_idx = static_cast<uint32_t>(events_.size());
        const uint32_t l_new_idx = r_new_idx + 1;

        const PolyId pol = events_[le_idx].pol;
        const uint8_t sp = events_[le_idx].subj_parity;
        const uint8_t cp = events_[le_idx].clip_parity;

        SweepEvent r_new{};
        r_new.point = ip;
        r_new.other_event = le_idx;
        r_new.pol = pol;
        r_new.left = false;
        r_new.subj_parity = sp;
        r_new.clip_parity = cp;

        SweepEvent l_new{};
        l_new.point = ip;
        l_new.other_event = old_r_idx;
        l_new.pol = pol;
        l_new.left = true;
        l_new.subj_parity = sp;
        l_new.clip_parity = cp;

        events_[old_r_idx].other_event = l_new_idx;
        events_[le_idx].other_event = r_new_idx;

        events_.push_back(r_new);
        events_.push_back(l_new);
        ensure_sl_capacity(l_new_idx);

        if (was_in_sl) {
            auto [it, inserted] = sl_.insert(le_idx);
            if (inserted) {
                events_[le_idx].in_sl = true;
                sl_its_[le_idx] = it;
            }
        }

        eq_.push(r_new_idx);
        eq_.push(l_new_idx);

        if (ref_idx != kBoolInvalidIdx && points_equal(ip, events_[ref_idx].point)) {
            // Since r_new is a right event at ref_idx's point (which is a left event),
            // r_new always precedes ref_idx in sweep order, so ref_idx must re-evaluate.
            return true;
        }
        return false;
    }

    /**
     * @brief Merges collinear segment `victim_idx` (identical endpoints) into `keeper_idx`.
     */
    void merge_identical_segments(uint32_t keeper_idx, uint32_t victim_idx) noexcept {
        auto& keeper = events_[keeper_idx];
        auto& victim = events_[victim_idx];

        keeper.subj_parity ^= victim.subj_parity;
        keeper.clip_parity ^= victim.clip_parity;

        if (victim.in_sl) {
            sl_.erase(sl_its_[victim_idx]);
            victim.in_sl = false;
        }
        victim.dead = true;
        events_[victim.other_event].dead = true;

        if (keeper.subj_parity == 0 && keeper.clip_parity == 0) {
            if (keeper.in_sl) {
                sl_.erase(sl_its_[keeper_idx]);
                keeper.in_sl = false;
            }
            keeper.dead = true;
            events_[keeper.other_event].dead = true;
        } else {
            recompute_fields_in_sl(keeper_idx);
        }
    }

    /**
     * @brief Detects and resolves crossings or collinear overlaps between two segments in `sl_`.
     * @return true if `cur_idx` needs re-queuing because a neighbor was split at `cur_idx.point`.
     */
    bool handle_intersection(uint32_t e1_idx, uint32_t e2_idx, uint32_t cur_idx) {
        if (e1_idx == e2_idx || events_[e1_idx].dead || events_[e2_idx].dead) {
            return false;
        }

        const Point2D l1 = events_[e1_idx].point;
        const Point2D r1 = events_[events_[e1_idx].other_event].point;
        const Point2D l2 = events_[e2_idx].point;
        const Point2D r2 = events_[events_[e2_idx].other_event].point;

        Point2D ip1{};
        Point2D ip2{};
        const int ninter = segment_intersection(l1, r1, l2, r2, ip1, ip2);
        if (ninter == 0) {
            return false;
        }

        if (ninter == 1) {
            const bool on_e1_end = points_equal(ip1, l1) || points_equal(ip1, r1);
            const bool on_e2_end = points_equal(ip1, l2) || points_equal(ip1, r2);
            if (on_e1_end && on_e2_end) {
                return false;
            }
            bool requeue = false;
            if (!on_e1_end) {
                requeue |= divide_segment(e1_idx, ip1, cur_idx);
            }
            if (!on_e2_end) {
                requeue |= divide_segment(e2_idx, ip1, cur_idx);
            }
            return requeue;
        }

        // ninter == 2: Collinear overlap!
        if (points_equal(l1, l2) && points_equal(r1, r2)) {
            if (e1_idx == cur_idx) {
                merge_identical_segments(e2_idx, e1_idx);
            } else {
                merge_identical_segments(e1_idx, e2_idx);
            }
            return false;
        }

        if (points_equal(l1, l2)) {
            // Shared left endpoint, different right endpoints: split longer segment at shorter R
            bool requeue = false;
            if (point_less(r1, r2)) {
                requeue = divide_segment(e2_idx, r1, cur_idx);
            } else {
                requeue = divide_segment(e1_idx, r2, cur_idx);
            }
            if (!events_[e1_idx].dead && !events_[e2_idx].dead &&
                points_equal(events_[events_[e1_idx].other_event].point,
                             events_[events_[e2_idx].other_event].point)) {
                merge_identical_segments(e1_idx, e2_idx);
            }
            return requeue;
        }

        // Different left endpoints: split the segment that started earlier at max(l1, l2)
        if (point_less(l1, l2)) {
            return divide_segment(e1_idx, l2, cur_idx);
        }
        return divide_segment(e2_idx, l1, cur_idx);
    }

    // ========================================================================
    // Phase 3: Leftmost-Turn Loop-Slicing & Area-Sorted Hole Nesting
    // ========================================================================

    struct HalfEdge {
        uint32_t to{0};
        uint32_t edge_id{0};
    };

    struct PlanarGraph {
        std::vector<Point2D> coords;
        std::vector<std::vector<HalfEdge>> adj;
        std::vector<bool> edge_used;
    };

    [[nodiscard]] static bool upper_half(const Point2D& u, const Point2D& v) noexcept {
        const double dy = v.y - u.y;
        const double dx = v.x - u.x;
        return (dy > 0.0) || (dy == 0.0 && dx > 0.0);
    }

    [[nodiscard]] static int point_in_ring_strict(const Point2D& pt,
                                                  std::span<const Point2D> ring) noexcept {
        // Hormann-Agathos / standard half-open even-odd ray casting to +infinity x using orient2d
        const std::size_t n = ring.size();
        bool inside = false;
        for (std::size_t i = 0; i < n; ++i) {
            const Point2D& a = ring[i];
            const Point2D& b = ring[(i + 1 == n) ? 0 : (i + 1)];
            if ((a.y > pt.y) != (b.y > pt.y)) {
                // Edge crosses horizontal line y = pt.y
                if (a.y < b.y) {
                    if (orient2d(a, b, pt) == Orientation::CounterClockwise) {
                        inside = !inside;
                    }
                } else {
                    if (orient2d(b, a, pt) == Orientation::CounterClockwise) {
                        inside = !inside;
                    }
                }
            }
        }
        return inside ? 1 : 0;
    }

    static void simplify_extracted_ring(std::vector<Point2D>& ring) {
        if (ring.size() < 3) {
            ring.clear();
            return;
        }
        // Remove collinear intermediate vertices and 180-degree spikes
        bool changed = true;
        while (changed && ring.size() >= 3) {
            changed = false;
            std::vector<Point2D> next;
            next.reserve(ring.size());
            const std::size_t n = ring.size();
            for (std::size_t i = 0; i < n; ++i) {
                const Point2D& prev = next.empty() ? ring[(i + n - 1) % n] : next.back();
                const Point2D& curr = ring[i];
                const Point2D& nxt = ring[(i + 1) % n];
                if (points_equal(curr, prev) || points_equal(curr, nxt) ||
                    points_equal(prev, nxt) ||
                    orient2d(prev, curr, nxt) == Orientation::Collinear) {
                    changed = true;
                    continue;
                }
                next.push_back(curr);
            }
            ring.swap(next);
        }
        if (ring.size() < 3 || std::abs(ring_signed_area<Point2D>(ring)) == 0.0) {
            ring.clear();
        }
    }

    /// @brief Builds the angularly-sorted planar subdivision graph from contributing result
    /// segments.
    [[nodiscard]] PlanarGraph build_planar_graph() const {
        struct SegEndpoints {
            Point2D a;
            Point2D b;
        };
        std::vector<SegEndpoints> segs;
        segs.reserve(result_segments_.size());

        PlanarGraph graph;
        graph.coords.reserve(result_segments_.size() * 2);

        for (uint32_t le_idx : result_segments_) {
            const auto& le = events_[le_idx];
            if (le.dead || !le.in_result) {
                continue;
            }
            const Point2D a = le.point;
            const Point2D b = events_[le.other_event].point;
            if (points_equal(a, b)) {
                continue;
            }
            segs.push_back({a, b});
            graph.coords.push_back(a);
            graph.coords.push_back(b);
        }
        if (segs.empty()) {
            return graph;
        }

        std::sort(graph.coords.begin(), graph.coords.end(), point_less);
        graph.coords.erase(std::unique(graph.coords.begin(), graph.coords.end(), points_equal),
                           graph.coords.end());

        auto find_vid = [&](const Point2D& p) noexcept -> uint32_t {
            auto it = std::lower_bound(graph.coords.begin(), graph.coords.end(), p, point_less);
            return static_cast<uint32_t>(it - graph.coords.begin());
        };

        const uint32_t num_verts = static_cast<uint32_t>(graph.coords.size());
        graph.adj.resize(num_verts);
        graph.edge_used.assign(segs.size(), false);

        for (uint32_t e_id = 0; e_id < static_cast<uint32_t>(segs.size()); ++e_id) {
            const uint32_t u = find_vid(segs[e_id].a);
            const uint32_t v = find_vid(segs[e_id].b);
            if (u == v) {
                graph.edge_used[e_id] = true;
                continue;
            }
            graph.adj[u].push_back({v, e_id});
            graph.adj[v].push_back({u, e_id});
        }

        // Sort outgoing half-edges at each vertex in strict CCW angular order
        for (uint32_t u = 0; u < num_verts; ++u) {
            auto& list = graph.adj[u];
            if (list.size() <= 1) {
                continue;
            }
            const Point2D& pu = graph.coords[u];
            std::sort(list.begin(), list.end(), [&](const HalfEdge& ha, const HalfEdge& hb) {
                if (ha.to == hb.to) {
                    return ha.edge_id < hb.edge_id;
                }
                const Point2D& pa = graph.coords[ha.to];
                const Point2D& pb = graph.coords[hb.to];
                const bool ua = upper_half(pu, pa);
                const bool ub = upper_half(pu, pb);
                if (ua != ub) {
                    return ua;
                }
                const Orientation o = orient2d(pu, pa, pb);
                if (o != Orientation::Collinear) {
                    return o == Orientation::CounterClockwise;
                }
                return ha.edge_id < hb.edge_id;
            });
        }

        return graph;
    }

    /// @brief Extracts simple closed rings from `graph` via sharpest-left-turn cycle slicing.
    [[nodiscard]] static std::vector<std::vector<Point2D>> extract_raw_rings(PlanarGraph& graph) {
        const uint32_t num_verts = static_cast<uint32_t>(graph.coords.size());
        std::vector<std::vector<Point2D>> raw_rings;
        std::vector<std::size_t> vert_stack_pos(num_verts, std::numeric_limits<std::size_t>::max());
        std::vector<uint32_t> stack_verts;

        auto pick_next_edge = [&](uint32_t u, uint32_t prev_v, bool has_prev) noexcept -> uint32_t {
            const auto& list = graph.adj[u];
            const std::size_t deg = list.size();
            if (deg == 0) {
                return kBoolInvalidIdx;
            }
            if (!has_prev) {
                for (std::size_t i = 0; i < deg; ++i) {
                    if (!graph.edge_used[list[i].edge_id]) {
                        return static_cast<uint32_t>(i);
                    }
                }
                return kBoolInvalidIdx;
            }
            // Locate incoming reverse edge (u -> prev_v) and step CW (sharpest left turn)
            std::size_t in_idx = 0;
            for (std::size_t i = 0; i < deg; ++i) {
                if (list[i].to == prev_v) {
                    in_idx = i;
                    break;
                }
            }
            for (std::size_t step = 1; step <= deg; ++step) {
                const std::size_t cand = (in_idx + deg - step) % deg;
                if (!graph.edge_used[list[cand].edge_id]) {
                    return static_cast<uint32_t>(cand);
                }
            }
            return kBoolInvalidIdx;
        };

        for (uint32_t start_v = 0; start_v < num_verts; ++start_v) {
            while (true) {
                const uint32_t first_idx = pick_next_edge(start_v, 0, false);
                if (first_idx == kBoolInvalidIdx) {
                    break;
                }
                stack_verts.clear();
                vert_stack_pos[start_v] = 0;
                stack_verts.push_back(start_v);

                uint32_t cur_v = start_v;
                uint32_t prev_v = 0;
                bool has_prev = false;

                while (true) {
                    const uint32_t out_idx =
                        has_prev ? pick_next_edge(cur_v, prev_v, true) : first_idx;
                    if (out_idx == kBoolInvalidIdx) {
                        for (uint32_t v : stack_verts) {
                            vert_stack_pos[v] = std::numeric_limits<std::size_t>::max();
                        }
                        break;
                    }
                    const auto& he = graph.adj[cur_v][out_idx];
                    graph.edge_used[he.edge_id] = true;
                    const uint32_t nxt_v = he.to;

                    if (vert_stack_pos[nxt_v] != std::numeric_limits<std::size_t>::max()) {
                        // Slice off closed simple cycle [pos .. end]
                        const std::size_t pos = vert_stack_pos[nxt_v];
                        std::vector<Point2D> ring;
                        ring.reserve(stack_verts.size() - pos);
                        for (std::size_t k = pos; k < stack_verts.size(); ++k) {
                            ring.push_back(graph.coords[stack_verts[k]]);
                            if (k > pos) {
                                vert_stack_pos[stack_verts[k]] =
                                    std::numeric_limits<std::size_t>::max();
                            }
                        }
                        stack_verts.resize(pos + 1);
                        simplify_extracted_ring(ring);
                        if (!ring.empty()) {
                            raw_rings.push_back(std::move(ring));
                        }
                        if (pos == 0) {
                            vert_stack_pos[start_v] = std::numeric_limits<std::size_t>::max();
                            break;
                        }
                        cur_v = stack_verts.back();
                        has_prev = (stack_verts.size() >= 2);
                        prev_v = has_prev ? stack_verts[stack_verts.size() - 2] : 0;
                    } else {
                        vert_stack_pos[nxt_v] = stack_verts.size();
                        stack_verts.push_back(nxt_v);
                        prev_v = cur_v;
                        cur_v = nxt_v;
                        has_prev = true;
                    }
                }
            }
        }

        return raw_rings;
    }

    /// @brief Nests extracted rings by descending area and emits CCW outers with CW holes.
    template <Point2DLike PointType>
    [[nodiscard]] static MultiPolygon<PointType>
    assemble_multipolygon(std::vector<std::vector<Point2D>>& raw_rings, double cx, double cy) {
        struct RingInfo {
            std::vector<Point2D> pts;
            double abs_area{0.0};
            double signed_area{0.0};
            double min_x{0.0}, max_x{0.0}, min_y{0.0}, max_y{0.0};
            Point2D probe_pt{};
            int parent{-1};
            int depth{0};
            int output_poly_idx{-1};
        };

        std::vector<RingInfo> infos;
        infos.reserve(raw_rings.size());
        for (auto& r : raw_rings) {
            RingInfo info;
            info.signed_area = ring_signed_area<Point2D>(r);
            info.abs_area = std::abs(info.signed_area);
            if (info.abs_area == 0.0 || r.size() < 3) {
                continue;
            }
            info.min_x = info.max_x = r[0].x;
            info.min_y = info.max_y = r[0].y;
            for (const auto& p : r) {
                info.min_x = std::min(info.min_x, p.x);
                info.max_x = std::max(info.max_x, p.x);
                info.min_y = std::min(info.min_y, p.y);
                info.max_y = std::max(info.max_y, p.y);
            }
            // Midpoint of first edge cannot lie on any other ring's edge after overlap merge
            info.probe_pt = {0.5 * (r[0].x + r[1].x), 0.5 * (r[0].y + r[1].y)};
            info.pts = std::move(r);
            infos.push_back(std::move(info));
        }

        std::sort(infos.begin(), infos.end(),
                  [](const RingInfo& a, const RingInfo& b) { return a.abs_area > b.abs_area; });

        for (std::size_t i = 0; i < infos.size(); ++i) {
            const auto& child = infos[i];
            for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                const auto& cand = infos[static_cast<std::size_t>(j)];
                if (child.probe_pt.x < cand.min_x || child.probe_pt.x > cand.max_x ||
                    child.probe_pt.y < cand.min_y || child.probe_pt.y > cand.max_y) {
                    continue;
                }
                if (point_in_ring_strict(child.probe_pt, cand.pts) != 0) {
                    infos[i].parent = j;
                    infos[i].depth = cand.depth + 1;
                    break;
                }
            }
        }

        MultiPolygon<PointType> result;
        for (auto& info : infos) {
            const bool is_hole = (info.depth & 1) != 0;
            const bool want_ccw = !is_hole;
            const bool is_ccw = info.signed_area > 0.0;
            if (is_ccw != want_ccw) {
                std::reverse(info.pts.begin(), info.pts.end());
            }

            std::vector<PointType> out_ring;
            out_ring.reserve(info.pts.size());
            for (const auto& p : info.pts) {
                out_ring.push_back(make_point<PointType>(p.x + cx, p.y + cy));
            }

            if (!is_hole) {
                info.output_poly_idx = static_cast<int>(result.size());
                BooleanPolygon<PointType> poly;
                poly.outer_ring = std::move(out_ring);
                result.push_back(std::move(poly));
            } else if (info.parent >= 0) {
                const int poly_idx = infos[static_cast<std::size_t>(info.parent)].output_poly_idx;
                if (poly_idx >= 0 && static_cast<std::size_t>(poly_idx) < result.size()) {
                    result[static_cast<std::size_t>(poly_idx)].holes.push_back(std::move(out_ring));
                }
            }
        }

        return result;
    }

    template <Point2DLike PointType>
    [[nodiscard]] MultiPolygon<PointType> reconstruct_multipolygon(double cx, double cy) {
        if (result_segments_.empty()) {
            return {};
        }

        PlanarGraph graph = build_planar_graph();
        if (graph.edge_used.empty()) {
            return {};
        }

        std::vector<std::vector<Point2D>> raw_rings = extract_raw_rings(graph);
        if (raw_rings.empty()) {
            return {};
        }

        return assemble_multipolygon<PointType>(raw_rings, cx, cy);
    }
};

} // namespace detail

/**
 * @brief Computes a Boolean set operation between two 2D polygons.
 *
 * @tparam PointType 2D point type satisfying @c Point2DLike.
 * @param subject First operand polygon.
 * @param clipping Second operand polygon.
 * @param op Boolean operation (Intersection, Union, Difference, or Xor).
 * @return Resulting multi-polygon containing zero or more outer rings with holes.
 */
template <Point2DLike PointType = std::array<double, 2>>
[[nodiscard]] MultiPolygon<PointType> polygon_boolean(const BooleanPolygon<PointType>& subject,
                                                      const BooleanPolygon<PointType>& clipping,
                                                      BooleanOp op) {
    const bool subj_empty = subject.outer_ring.size() < 3;
    const bool clip_empty = clipping.outer_ring.size() < 3;

    if (subj_empty && clip_empty) {
        return {};
    }

    // Compute translation origin for Sterbenz cancellation avoidance
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    auto accumulate_bbox = [&](const BooleanPolygon<PointType>& poly) {
        auto acc_ring = [&](std::span<const PointType> ring) {
            for (const auto& p : ring) {
                const double x = detail::get_x(p);
                const double y = detail::get_y(p);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        };
        acc_ring(poly.outer_ring);
        for (const auto& h : poly.holes) {
            acc_ring(h);
        }
    };

    if (!subj_empty) {
        accumulate_bbox(subject);
    }
    if (!clip_empty) {
        accumulate_bbox(clipping);
    }

    auto choose_origin = [](double lo, double hi) noexcept -> double {
        if (lo > 0.0 && hi > 0.0 && lo > (hi - lo)) {
            return lo;
        }
        if (lo < 0.0 && hi < 0.0 && (-hi) > (hi - lo)) {
            return hi;
        }
        return 0.0;
    };

    const double cx = (min_x <= max_x) ? choose_origin(min_x, max_x) : 0.0;
    const double cy = (min_y <= max_y) ? choose_origin(min_y, max_y) : 0.0;

    // Empty-operand fast paths still normalize self-intersections/holes if needed
    if (subj_empty && op == BooleanOp::Intersection) {
        return {};
    }
    if (subj_empty && op == BooleanOp::Difference) {
        return {};
    }
    if (clip_empty && op == BooleanOp::Intersection) {
        return {};
    }

    detail::SweepEngine engine(op);

    auto ingest_poly = [&](const BooleanPolygon<PointType>& poly, detail::PolyId pid) {
        if (poly.outer_ring.size() < 3) {
            return;
        }
        auto outer = detail::sanitize_ring<PointType>(poly.outer_ring, cx, cy);
        if (outer.size() >= 3) {
            engine.add_ring(outer, pid);
        }
        for (const auto& h : poly.holes) {
            auto hole = detail::sanitize_ring<PointType>(h, cx, cy);
            if (hole.size() >= 3) {
                engine.add_ring(hole, pid);
            }
        }
    };

    ingest_poly(subject, detail::PolyId::Subject);
    ingest_poly(clipping, detail::PolyId::Clipping);

    return engine.run<PointType>(cx, cy);
}

/// @brief Computes the geometric intersection (`a & b`) of two 2D polygons.
template <Point2DLike PointType = std::array<double, 2>>
[[nodiscard]] inline MultiPolygon<PointType>
polygon_intersection(const BooleanPolygon<PointType>& a, const BooleanPolygon<PointType>& b) {
    return polygon_boolean(a, b, BooleanOp::Intersection);
}

/// @brief Computes the geometric union (`a | b`) of two 2D polygons.
template <Point2DLike PointType = std::array<double, 2>>
[[nodiscard]] inline MultiPolygon<PointType> polygon_union(const BooleanPolygon<PointType>& a,
                                                           const BooleanPolygon<PointType>& b) {
    return polygon_boolean(a, b, BooleanOp::Union);
}

/// @brief Computes the geometric set difference (`a - b`) of two 2D polygons.
template <Point2DLike PointType = std::array<double, 2>>
[[nodiscard]] inline MultiPolygon<PointType>
polygon_difference(const BooleanPolygon<PointType>& a, const BooleanPolygon<PointType>& b) {
    return polygon_boolean(a, b, BooleanOp::Difference);
}

/// @brief Computes the symmetric difference (`a ^ b`) of two 2D polygons.
template <Point2DLike PointType = std::array<double, 2>>
[[nodiscard]] inline MultiPolygon<PointType> polygon_xor(const BooleanPolygon<PointType>& a,
                                                         const BooleanPolygon<PointType>& b) {
    return polygon_boolean(a, b, BooleanOp::Xor);
}

namespace boolean {

template <Point2DLike PointType = std::array<double, 2>> using Polygon = BooleanPolygon<PointType>;

#ifndef ALGOAT_BUILDING_MODULE
using algoat::geometry::BooleanOp;
using algoat::geometry::BooleanPolygon;
using algoat::geometry::MultiPolygon;
using algoat::geometry::Point2DLike;
using algoat::geometry::polygon_boolean;
using algoat::geometry::polygon_difference;
using algoat::geometry::polygon_intersection;
using algoat::geometry::polygon_union;
using algoat::geometry::polygon_xor;
#endif

} // namespace boolean

} // namespace algoat::geometry
