/**
 * @file bentley_ottmann.hpp
 * @brief Output-sensitive Bentley-Ottmann Sweep-Line Line Segment Intersection Engine.
 *
 * Implements the plane-sweep algorithm of Bentley & Ottmann (1979) with exact Shewchuk
 * adaptive orientation predicates, a cache-aligned 4-ary indexed min-heap event queue
 * (`algoat::graph::DAryHeap<4>`), and an array-backed threaded treap status structure.
 *
 * References:
 * - Bentley, J. L., & Ottmann, T. A. (1979). "Algorithms for reporting and counting
 *   geometric intersections." IEEE Transactions on Computers, C-28(9), 643-647.
 * - de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). "Computational
 *   Geometry: Algorithms and Applications" (3rd ed.), Chapter 2.
 * - Preparata, F. P., & Shamos, M. I. (1985). "Computational Geometry: An Introduction",
 *   Section 7.2.3; Boissonnat, J.-D., & Preparata, F. P. (2000), SIAM J. Comput. 29(5).
 */

#pragma once

#include "algoat/geometry/predicates.hpp"
#include "algoat/graph/d_ary_heap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::geometry {

/**
 * @struct LineSegment
 * @brief Closed 2D line segment defined by two endpoints in coordinate type @p T.
 */
template <typename T = double> struct LineSegment {
    static_assert(std::is_arithmetic_v<T>, "LineSegment coordinate type T must be arithmetic");
    using value_type = T;

    std::array<T, 2> start{};
    std::array<T, 2> end{};

    [[nodiscard]] constexpr bool operator==(const LineSegment&) const noexcept = default;
};

/**
 * @struct IntersectionResult
 * @brief Pairwise segment intersection record reporting intersection coordinates and
 *        original segment indices with @c segment1_index < @c segment2_index.
 */
template <typename T = double> struct IntersectionResult {
    static_assert(std::is_arithmetic_v<T>,
                  "IntersectionResult coordinate type T must be arithmetic");
    using value_type = T;

    std::array<T, 2> point{};
    std::size_t segment1_index{0};
    std::size_t segment2_index{0};

    [[nodiscard]] constexpr bool operator==(const IntersectionResult&) const noexcept = default;
};

namespace detail {

inline constexpr uint32_t kBoInvalid = 0xFFFFFFFFu;

template <typename R>
concept IsLineSegmentRange = requires {
    typename std::remove_cvref_t<R>::value_type;
    requires std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R>>,
                          LineSegment<typename std::remove_cvref_t<R>::value_type::value_type>>;
} && std::ranges::contiguous_range<R>;

enum class VerticalRelation : int8_t { Below = -1, Coincident = 0, Above = 1 };

struct CanonSegment {
    Point2D p1{};
    Point2D p2{};
    double dx{0.0};
    double dy{0.0};
    double min_y{0.0};
    double max_y{0.0};
    uint32_t orig_index{0};
    bool is_vertical{false};
    bool is_point{false};
};

[[nodiscard]] constexpr bool bo_point_less(const Point2D& a, const Point2D& b) noexcept {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
}

[[nodiscard]] constexpr bool bo_points_equal(const Point2D& a, const Point2D& b) noexcept {
    return a.x == b.x && a.y == b.y;
}

[[nodiscard]] constexpr bool bo_point_in_box(const Point2D& p, const Point2D& a,
                                             const Point2D& b) noexcept {
    const double min_x = std::min(a.x, b.x);
    const double max_x = std::max(a.x, b.x);
    const double min_y = std::min(a.y, b.y);
    const double max_y = std::max(a.y, b.y);
    return p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y;
}

inline void bo_two_diff_exact(double a, double b, double& hi, double& lo) noexcept {
    hi = a - b;
    const double bvirt = a - hi;
    const double avirt = hi + bvirt;
    const double bround = bvirt - b;
    const double around = a - avirt;
    lo = around + bround;
}

inline void expand_two_by_two_product(const double* a2, const double* b2, double* prod8,
                                      int& prod_len) noexcept {
    double p00_hi = 0.0, p00_lo = 0.0;
    double p01_hi = 0.0, p01_lo = 0.0;
    double p10_hi = 0.0, p10_lo = 0.0;
    double p11_hi = 0.0, p11_lo = 0.0;
    two_product_fma(a2[0], b2[0], p00_hi, p00_lo);
    two_product_fma(a2[0], b2[1], p01_hi, p01_lo);
    two_product_fma(a2[1], b2[0], p10_hi, p10_lo);
    two_product_fma(a2[1], b2[1], p11_hi, p11_lo);

    double e0[2] = {p00_lo, p00_hi};
    double e1[2] = {p01_lo, p01_hi};
    double e2[2] = {p10_lo, p10_hi};
    double e3[2] = {p11_lo, p11_hi};

    double s01[4]{};
    const int l01 = fast_expansion_sum_zeroelim(2, e0, 2, e1, s01);
    double s23[4]{};
    const int l23 = fast_expansion_sum_zeroelim(2, e2, 2, e3, s23);
    prod_len = fast_expansion_sum_zeroelim(l01, s01, l23, s23, prod8);
}

[[nodiscard]] inline int bo_max_exp(std::initializer_list<double> vals) noexcept {
    int max_e = std::numeric_limits<int>::min();
    for (double v : vals) {
        if (v != 0.0) {
            int e = 0;
            (void)std::frexp(v, &e);
            if (e > max_e) {
                max_e = e;
            }
        }
    }
    return (max_e == std::numeric_limits<int>::min()) ? 0 : max_e;
}

[[nodiscard]] inline int bo_expansion_sign(int len, const double* e) noexcept {
    if (len <= 0) {
        return 0;
    }
    const double msb = e[len - 1];
    if (msb > 0.0) {
        return 1;
    }
    if (msb < 0.0) {
        return -1;
    }
    return 0;
}

/**
 * @brief Evaluates exact position of non-vertical segment @p s relative to @p query.
 */
[[nodiscard]] inline VerticalRelation orient_seg_point(const CanonSegment& s,
                                                       const Point2D& query) noexcept {
    const Orientation o = orient2d(s.p1, s.p2, query);
    if (o == Orientation::CounterClockwise) {
        return VerticalRelation::Below;
    }
    if (o == Orientation::Clockwise) {
        return VerticalRelation::Above;
    }
    return VerticalRelation::Coincident;
}

/**
 * @brief Evaluates exact comparison of slopes dy_a / dx_a vs dy_b / dx_b for dx_a > 0, dx_b > 0.
 * @return -1 if slope(a) < slope(b), 0 if equal, +1 if slope(a) > slope(b).
 */
[[nodiscard]] inline int exact_slope_cmp(const CanonSegment& a, const CanonSegment& b) noexcept {
    if (std::isfinite(a.dx) && std::isfinite(a.dy) && std::isfinite(b.dx) && std::isfinite(b.dy)) {
        const double left = a.dy * b.dx;
        const double right = b.dy * a.dx;
        const double det = left - right;
        if (std::isfinite(left) && std::isfinite(right)) {
            const double mag = std::abs(left) + std::abs(right);
            if (mag == 0.0) {
                return 0;
            }
            if (mag > 1e-200 && mag < 1e200 &&
                std::abs(det) > 32.0 * std::numeric_limits<double>::epsilon() * mag) {
                return (det > 0.0) ? 1 : -1;
            }
        }
    }

    const int ea = bo_max_exp({a.p1.x, a.p2.x, a.p1.y, a.p2.y});
    const int eb = bo_max_exp({b.p1.x, b.p2.x, b.p1.y, b.p2.y});
    const double ax1 = std::ldexp(a.p1.x, -ea);
    const double ax2 = std::ldexp(a.p2.x, -ea);
    const double ay1 = std::ldexp(a.p1.y, -ea);
    const double ay2 = std::ldexp(a.p2.y, -ea);
    const double bx1 = std::ldexp(b.p1.x, -eb);
    const double bx2 = std::ldexp(b.p2.x, -eb);
    const double by1 = std::ldexp(b.p1.y, -eb);
    const double by2 = std::ldexp(b.p2.y, -eb);

    double adx[2]{}, ady[2]{}, bdx[2]{}, bdy[2]{};
    bo_two_diff_exact(ax2, ax1, adx[1], adx[0]);
    bo_two_diff_exact(ay2, ay1, ady[1], ady[0]);
    bo_two_diff_exact(bx2, bx1, bdx[1], bdx[0]);
    bo_two_diff_exact(by2, by1, bdy[1], bdy[0]);

    double t1[8]{}, t2[8]{};
    int l1 = 0, l2 = 0;
    expand_two_by_two_product(ady, bdx, t1, l1);
    const double neg_bdy[2] = {-bdy[0], -bdy[1]};
    expand_two_by_two_product(neg_bdy, adx, t2, l2);

    double res[16]{};
    const int rlen = fast_expansion_sum_zeroelim(l1, t1, l2, t2, res);
    return bo_expansion_sign(rlen, res);
}

/**
 * @brief Evaluates y(x) on non-vertical segment @p s safely even if endpoints span +-DBL_MAX.
 */
[[nodiscard]] inline double eval_y(const CanonSegment& s, double x) noexcept {
    if (x <= s.p1.x) {
        return s.p1.y;
    }
    if (x >= s.p2.x) {
        return s.p2.y;
    }
    if (!std::isfinite(s.dx) || !std::isfinite(s.dy)) {
        const double h_dx = std::ldexp(s.p2.x, -1) - std::ldexp(s.p1.x, -1);
        const double h_dy = std::ldexp(s.p2.y, -1) - std::ldexp(s.p1.y, -1);
        const double t = (std::ldexp(x, -1) - std::ldexp(s.p1.x, -1)) / h_dx;
        return std::ldexp(std::fma(t, h_dy, std::ldexp(s.p1.y, -1)), 1);
    }
    if ((x - s.p1.x) <= (s.p2.x - x)) {
        const double t = (x - s.p1.x) / s.dx;
        return std::fma(t, s.dy, s.p1.y);
    }
    const double t = (s.p2.x - x) / s.dx;
    return std::fma(-t, s.dy, s.p2.y);
}

[[nodiscard]] inline double band_tol(const CanonSegment& s, double y_ref) noexcept {
    const double dy_mag = std::isfinite(s.dy) ? std::abs(s.dy) : 0.0;
    const double scale =
        std::max({1.0, std::abs(s.p1.y), std::abs(s.p2.y), std::abs(y_ref), dy_mag});
    return 128.0 * std::numeric_limits<double>::epsilon() * scale;
}

/**
 * @brief Exact IEEE-754 order-preserving bit midpoint between finite @p lo < @p hi.
 */
[[nodiscard]] inline double bit_midpoint_double(double lo, double hi) noexcept {
    auto to_ordered = [](double v) noexcept -> uint64_t {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        constexpr uint64_t kSignBit = 1ULL << 63;
        if ((bits & kSignBit) != 0) {
            return kSignBit - (bits & ~kSignBit);
        }
        return kSignBit + bits;
    };
    auto from_ordered = [](uint64_t o) noexcept -> double {
        constexpr uint64_t kSignBit = 1ULL << 63;
        const uint64_t bits = (o < kSignBit) ? (kSignBit | (kSignBit - o)) : (o - kSignBit);
        double v = 0.0;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    const uint64_t oa = to_ordered(lo);
    const uint64_t ob = to_ordered(hi);
    return from_ordered(oa + ((ob - oa) >> 1));
}

struct SwapSchedule {
    double x{0.0};
    bool at_x{false};
};

[[nodiscard]] inline SwapSchedule bisect_crossing_x(const CanonSegment& a, const CanonSegment& b,
                                                    double lo, double hi) noexcept {
    while (std::nextafter(lo, hi) < hi) {
        const double mid = bit_midpoint_double(lo, hi);
        if (eval_y(a, mid) <= eval_y(b, mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return {lo, eval_y(a, lo) == eval_y(b, lo)};
}

/**
 * @brief Computes sweep x where eval_y(a, x) transitions from <= eval_y(b, x) to > eval_y(b, x).
 */
[[nodiscard]] inline SwapSchedule compute_swap_schedule(const CanonSegment& a,
                                                        const CanonSegment& b, double ip_x,
                                                        double x_sweep) noexcept {
    const double x_end = std::min(a.p2.x, b.p2.x);
    if (x_sweep >= x_end) {
        return {x_end, eval_y(a, x_end) == eval_y(b, x_end)};
    }

    const double ya_sw = eval_y(a, x_sweep);
    const double yb_sw = eval_y(b, x_sweep);
    if (ya_sw > yb_sw) {
        return {x_sweep, false};
    }

    const double ya_end = eval_y(a, x_end);
    const double yb_end = eval_y(b, x_end);
    if (ya_end <= yb_end) {
        return {x_end, ya_end == yb_end};
    }

    const double x_clamped = std::clamp(ip_x, x_sweep, x_end);
    const double ya_c = eval_y(a, x_clamped);
    const double yb_c = eval_y(b, x_clamped);
    if (ya_c == yb_c) {
        return {x_clamped, true};
    }

    if (ya_c < yb_c) {
        const double x_next = std::nextafter(x_clamped, x_end);
        if (x_next > x_end || eval_y(a, x_next) > eval_y(b, x_next)) {
            return {x_clamped, false};
        }
        return bisect_crossing_x(a, b, x_next, x_end);
    }

    const double x_prev = std::nextafter(x_clamped, x_sweep);
    if (x_prev <= x_sweep) {
        return {x_sweep, ya_sw == yb_sw};
    }
    if (eval_y(a, x_prev) <= eval_y(b, x_prev)) {
        return {x_prev, eval_y(a, x_prev) == eval_y(b, x_prev)};
    }
    return bisect_crossing_x(a, b, x_sweep, x_prev);
}

/**
 * @brief Tests exact closed-segment intersection and returns canonical intersection coordinates.
 */
[[nodiscard]] inline bool intersect_segments_exact(const CanonSegment& s1, const CanonSegment& s2,
                                                   Point2D& ip) noexcept {
    if (s1.p2.x < s2.p1.x || s2.p2.x < s1.p1.x || s1.max_y < s2.min_y || s2.max_y < s1.min_y) {
        return false;
    }

    // Canonicalize pair order for bit-identical intersection point coordinates
    const CanonSegment* pa = &s1;
    const CanonSegment* pb = &s2;
    if (bo_point_less(s2.p1, s1.p1) ||
        (bo_points_equal(s1.p1, s2.p1) && bo_point_less(s2.p2, s1.p2))) {
        std::swap(pa, pb);
    }

    const Point2D& a0 = pa->p1;
    const Point2D& a1 = pa->p2;
    const Point2D& b0 = pb->p1;
    const Point2D& b1 = pb->p2;

    if (pa->is_point && pb->is_point) {
        if (bo_points_equal(a0, b0)) {
            ip = a0;
            return true;
        }
        return false;
    }
    if (pa->is_point) {
        if (bo_point_in_box(a0, b0, b1) && orient2d(b0, b1, a0) == Orientation::Collinear) {
            ip = a0;
            return true;
        }
        return false;
    }
    if (pb->is_point) {
        if (bo_point_in_box(b0, a0, a1) && orient2d(a0, a1, b0) == Orientation::Collinear) {
            ip = b0;
            return true;
        }
        return false;
    }

    const Orientation op0 = orient2d(a0, a1, b0);
    const Orientation op1 = orient2d(a0, a1, b1);
    const Orientation oq0 = orient2d(b0, b1, a0);
    const Orientation oq1 = orient2d(b0, b1, a1);

    // Collinear overlap: lexicographically earliest shared point
    if (op0 == Orientation::Collinear && op1 == Orientation::Collinear) {
        const Point2D lo = bo_point_less(a0, b0) ? b0 : a0;
        const Point2D hi = bo_point_less(a1, b1) ? a1 : b1;
        if (bo_point_less(hi, lo)) {
            return false;
        }
        ip = lo;
        return true;
    }

    // Shared endpoints & collinear T-junctions
    if (bo_points_equal(a0, b0) || bo_points_equal(a0, b1)) {
        ip = a0;
        return true;
    }
    if (bo_points_equal(a1, b0) || bo_points_equal(a1, b1)) {
        ip = a1;
        return true;
    }
    if (op0 == Orientation::Collinear && bo_point_in_box(b0, a0, a1)) {
        ip = b0;
        return true;
    }
    if (op1 == Orientation::Collinear && bo_point_in_box(b1, a0, a1)) {
        ip = b1;
        return true;
    }
    if (oq0 == Orientation::Collinear && bo_point_in_box(a0, b0, b1)) {
        ip = a0;
        return true;
    }
    if (oq1 == Orientation::Collinear && bo_point_in_box(a1, b0, b1)) {
        ip = a1;
        return true;
    }

    if ((op0 == op1 && op0 != Orientation::Collinear) ||
        (oq0 == oq1 && oq0 != Orientation::Collinear)) {
        return false;
    }

    // Proper crossing: scale coordinates if extreme magnitudes risk overflow/underflow
    const double min_x = std::max(pa->p1.x, pb->p1.x);
    const double max_x = std::min(pa->p2.x, pb->p2.x);
    const double min_y = std::max(pa->min_y, pb->min_y);
    const double max_y = std::min(pa->max_y, pb->max_y);

    const double max_mag =
        std::max({std::abs(a0.x), std::abs(a0.y), std::abs(a1.x), std::abs(a1.y), std::abs(b0.x),
                  std::abs(b0.y), std::abs(b1.x), std::abs(b1.y)});
    int shift = 0;
    if (max_mag > 1e140 || (max_mag > 0.0 && max_mag < 1e-140)) {
        shift = bo_max_exp({a0.x, a0.y, a1.x, a1.y, b0.x, b0.y, b1.x, b1.y});
    }

    const double sa0x = std::ldexp(a0.x, -shift);
    const double sa0y = std::ldexp(a0.y, -shift);
    const double sa1x = std::ldexp(a1.x, -shift);
    const double sa1y = std::ldexp(a1.y, -shift);
    const double sb0x = std::ldexp(b0.x, -shift);
    const double sb0y = std::ldexp(b0.y, -shift);
    const double sb1x = std::ldexp(b1.x, -shift);
    const double sb1y = std::ldexp(b1.y, -shift);

    const double dx0 = sa1x - sa0x;
    const double dy0 = sa1y - sa0y;
    const double dx1 = sb1x - sb0x;
    const double dy1 = sb1y - sb0y;
    const double cross_det = std::fma(dx0, dy1, -dy0 * dx1);
    if (cross_det == 0.0 || !std::isfinite(cross_det)) {
        ip = Point2D{bit_midpoint_double(min_x, max_x), bit_midpoint_double(min_y, max_y)};
        return true;
    }

    const double ex = sb0x - sa0x;
    const double ey = sb0y - sa0y;
    const double t_a = std::clamp(std::fma(ex, dy1, -ey * dx1) / cross_det, 0.0, 1.0);
    const double t_b = std::clamp(std::fma(ex, dy0, -ey * dx0) / cross_det, 0.0, 1.0);

    const double cx = std::ldexp((std::abs(dx0) <= std::abs(dx1)) ? std::fma(t_a, dx0, sa0x)
                                                                  : std::fma(t_b, dx1, sb0x),
                                 shift);
    const double cy = std::ldexp((std::abs(dy0) <= std::abs(dy1)) ? std::fma(t_a, dy0, sa0y)
                                                                  : std::fma(t_b, dy1, sb0y),
                                 shift);

    ip = Point2D{std::clamp(cx, min_x, max_x), std::clamp(cy, min_y, max_y)};
    return true;
}

/**
 * @brief Flat open-addressed hash set of unordered segment index pairs (lo < hi).
 *
 * Uses @c kEmptyKey = 0 because valid pairs satisfy @c lo < @c hi, hence @c hi >= 1
 * and packed keys @c (lo << 32) | hi are strictly non-zero.
 */
class PairSet {
public:
    [[nodiscard]] bool insert(uint32_t a, uint32_t b) {
        const uint64_t key = pack_pair(a, b);
        if (table_.empty()) {
            rehash(64);
        } else if ((size_ + 1) * 10 >= table_.size() * 7) {
            rehash(table_.size() * 2);
        }

        const std::size_t mask = table_.size() - 1;
        std::size_t idx = hash_u64(key) & mask;
        while (true) {
            const uint64_t cur = table_[idx];
            if (cur == kEmptyKey) {
                table_[idx] = key;
                ++size_;
                return true;
            }
            if (cur == key) {
                return false;
            }
            idx = (idx + 1) & mask;
        }
    }

private:
    static constexpr uint64_t kEmptyKey = 0ULL;

    static constexpr uint64_t pack_pair(uint32_t a, uint32_t b) noexcept {
        const uint32_t lo = std::min(a, b);
        const uint32_t hi = std::max(a, b);
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    }

    static constexpr uint64_t hash_u64(uint64_t x) noexcept {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    void rehash(std::size_t new_cap) {
        std::vector<uint64_t> next(new_cap, kEmptyKey);
        const std::size_t mask = new_cap - 1;
        for (uint64_t k : table_) {
            if (k != kEmptyKey) {
                std::size_t idx = hash_u64(k) & mask;
                while (next[idx] != kEmptyKey) {
                    idx = (idx + 1) & mask;
                }
                next[idx] = k;
            }
        }
        table_.swap(next);
    }

    std::vector<uint64_t> table_;
    std::size_t size_{0};
};

/**
 * @brief Array-backed threaded treap maintaining active non-vertical segments.
 */
class StatusTreap {
public:
    struct Node {
        uint32_t left{kBoInvalid};
        uint32_t right{kBoInvalid};
        uint32_t parent{kBoInvalid};
        uint32_t prev{kBoInvalid};
        uint32_t next{kBoInvalid};
        uint32_t prio{0};
        bool in_status{false};
    };

    explicit StatusTreap(const std::vector<CanonSegment>& segs) : segs_(segs), nodes_(segs.size()) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(nodes_.size()); ++i) {
            nodes_[i].prio = splitmix32(i + 1u);
        }
    }

    [[nodiscard]] bool contains(uint32_t u) const noexcept {
        return u < nodes_.size() && nodes_[u].in_status;
    }

    [[nodiscard]] uint32_t prev_of(uint32_t u) const noexcept {
        return nodes_[u].prev;
    }

    [[nodiscard]] uint32_t next_of(uint32_t u) const noexcept {
        return nodes_[u].next;
    }

    void insert(uint32_t u, double x_sweep) {
        Node& nu = nodes_[u];
        nu.left = kBoInvalid;
        nu.right = kBoInvalid;
        nu.parent = kBoInvalid;
        nu.in_status = true;

        if (root_ == kBoInvalid) {
            root_ = u;
            nu.prev = kBoInvalid;
            nu.next = kBoInvalid;
            return;
        }

        const CanonSegment& su = segs_[u];
        uint32_t cur = root_;
        uint32_t pred = kBoInvalid;
        uint32_t succ = kBoInvalid;

        while (true) {
            const CanonSegment& sc = segs_[cur];
            const double yc = eval_y(sc, x_sweep);
            int cmp = 0;
            if (yc != su.p1.y) {
                cmp = (yc > su.p1.y) ? 1 : -1;
            } else {
                const VerticalRelation rel = orient_seg_point(sc, su.p1);
                cmp = static_cast<int>(rel);
                if (rel == VerticalRelation::Coincident) {
                    cmp = exact_slope_cmp(sc, su);
                    if (cmp == 0) {
                        cmp = (cur < u) ? -1 : 1;
                    }
                }
            }

            if (cmp > 0) {
                succ = cur;
                if (nodes_[cur].left == kBoInvalid) {
                    nodes_[cur].left = u;
                    nu.parent = cur;
                    break;
                }
                cur = nodes_[cur].left;
            } else {
                pred = cur;
                if (nodes_[cur].right == kBoInvalid) {
                    nodes_[cur].right = u;
                    nu.parent = cur;
                    break;
                }
                cur = nodes_[cur].right;
            }
        }

        nu.prev = pred;
        nu.next = succ;
        if (pred != kBoInvalid) {
            nodes_[pred].next = u;
        }
        if (succ != kBoInvalid) {
            nodes_[succ].prev = u;
        }

        while (nu.parent != kBoInvalid && nu.prio > nodes_[nu.parent].prio) {
            const uint32_t p = nu.parent;
            if (nodes_[p].left == u) {
                rotate_right(p);
            } else {
                rotate_left(p);
            }
        }

        while (nu.prev != kBoInvalid && eval_y(segs_[nu.prev], x_sweep) > su.p1.y) {
            swap_adjacent(nu.prev, u);
        }
        while (nu.next != kBoInvalid && eval_y(segs_[nu.next], x_sweep) < su.p1.y) {
            swap_adjacent(u, nu.next);
        }
    }

    void erase(uint32_t u) noexcept {
        Node& nu = nodes_[u];
        if (!nu.in_status) {
            return;
        }

        const uint32_t p = nu.prev;
        const uint32_t n = nu.next;
        if (p != kBoInvalid) {
            nodes_[p].next = n;
        }
        if (n != kBoInvalid) {
            nodes_[n].prev = p;
        }
        nu.prev = kBoInvalid;
        nu.next = kBoInvalid;

        while (nu.left != kBoInvalid || nu.right != kBoInvalid) {
            if (nu.left == kBoInvalid) {
                rotate_left(u);
            } else if (nu.right == kBoInvalid) {
                rotate_right(u);
            } else if (nodes_[nu.left].prio > nodes_[nu.right].prio) {
                rotate_right(u);
            } else {
                rotate_left(u);
            }
        }

        const uint32_t par = nu.parent;
        if (par == kBoInvalid) {
            root_ = kBoInvalid;
        } else if (nodes_[par].left == u) {
            nodes_[par].left = kBoInvalid;
        } else {
            nodes_[par].right = kBoInvalid;
        }
        nu.parent = kBoInvalid;
        nu.in_status = false;
    }

    void swap_adjacent(uint32_t a, uint32_t b) noexcept {
        const uint32_t p_thread = nodes_[a].prev;
        const uint32_t n_thread = nodes_[b].next;
        if (p_thread != kBoInvalid) {
            nodes_[p_thread].next = b;
        }
        nodes_[b].prev = p_thread;
        nodes_[b].next = a;
        nodes_[a].prev = b;
        nodes_[a].next = n_thread;
        if (n_thread != kBoInvalid) {
            nodes_[n_thread].prev = a;
        }

        const uint32_t al = nodes_[a].left;
        const uint32_t ar = nodes_[a].right;
        const uint32_t bl = nodes_[b].left;
        const uint32_t br = nodes_[b].right;
        const uint32_t pa = nodes_[a].parent;
        const uint32_t pb = nodes_[b].parent;

        if (pa == b) {
            replace_child_or_root(pb, b, a);
            attach_node(a, pb, b, br);
            attach_node(b, a, al, ar);
        } else if (pb == a) {
            replace_child_or_root(pa, a, b);
            attach_node(b, pa, al, a);
            attach_node(a, b, bl, br);
        } else {
            replace_child_or_root(pa, a, b);
            replace_child_or_root(pb, b, a);
            attach_node(a, pb, bl, br);
            attach_node(b, pa, al, ar);
        }
        std::swap(nodes_[a].prio, nodes_[b].prio);
    }

    [[nodiscard]] uint32_t find_vertical_lower_bound(double x0, double y_min) const noexcept {
        const Point2D q_min{x0, y_min};
        uint32_t cur = root_;
        uint32_t best = kBoInvalid;
        while (cur != kBoInvalid) {
            if (eval_y(segs_[cur], x0) < y_min &&
                orient_seg_point(segs_[cur], q_min) == VerticalRelation::Below) {
                cur = nodes_[cur].right;
            } else {
                best = cur;
                cur = nodes_[cur].left;
            }
        }
        if (best == kBoInvalid && root_ != kBoInvalid) {
            uint32_t rightmost = root_;
            while (nodes_[rightmost].right != kBoInvalid) {
                rightmost = nodes_[rightmost].right;
            }
            if (eval_y(segs_[rightmost], x0) >= y_min - band_tol(segs_[rightmost], y_min)) {
                best = rightmost;
            }
        }
        while (best != kBoInvalid) {
            const uint32_t p = nodes_[best].prev;
            if (p == kBoInvalid) {
                break;
            }
            if (orient_seg_point(segs_[p], q_min) == VerticalRelation::Below &&
                eval_y(segs_[p], x0) < y_min - band_tol(segs_[p], y_min)) {
                break;
            }
            best = p;
        }
        return best;
    }

private:
    static constexpr uint32_t splitmix32(uint32_t x) noexcept {
        x += 0x9e3779b9u;
        x = (x ^ (x >> 16)) * 0x85ebca6bu;
        x = (x ^ (x >> 13)) * 0xc2b2ae35u;
        return x ^ (x >> 16);
    }

    void attach_node(uint32_t node, uint32_t parent, uint32_t left, uint32_t right) noexcept {
        nodes_[node].parent = parent;
        nodes_[node].left = left;
        nodes_[node].right = right;
        if (left != kBoInvalid) {
            nodes_[left].parent = node;
        }
        if (right != kBoInvalid) {
            nodes_[right].parent = node;
        }
    }

    void replace_child_or_root(uint32_t p, uint32_t old_c, uint32_t new_c) noexcept {
        if (p == kBoInvalid) {
            root_ = new_c;
        } else if (nodes_[p].left == old_c) {
            nodes_[p].left = new_c;
        } else {
            nodes_[p].right = new_c;
        }
    }

    void rotate_left(uint32_t x) noexcept {
        const uint32_t y = nodes_[x].right;
        const uint32_t p = nodes_[x].parent;
        const uint32_t yl = nodes_[y].left;

        nodes_[x].right = yl;
        if (yl != kBoInvalid) {
            nodes_[yl].parent = x;
        }
        nodes_[y].parent = p;
        replace_child_or_root(p, x, y);
        nodes_[y].left = x;
        nodes_[x].parent = y;
    }

    void rotate_right(uint32_t x) noexcept {
        const uint32_t y = nodes_[x].left;
        const uint32_t p = nodes_[x].parent;
        const uint32_t yr = nodes_[y].right;

        nodes_[x].left = yr;
        if (yr != kBoInvalid) {
            nodes_[yr].parent = x;
        }
        nodes_[y].parent = p;
        replace_child_or_root(p, x, y);
        nodes_[y].right = x;
        nodes_[x].parent = y;
    }

    const std::vector<CanonSegment>& segs_;
    std::vector<Node> nodes_;
    uint32_t root_{kBoInvalid};
};

enum class EventType : uint8_t {
    LeftEndpoint = 0,
    RightEndpoint = 1,
    VerticalSegment = 2,
    IntersectionAtX = 3,
    IntersectionAfterX = 4
};

struct SweepEventRecord {
    double y{0.0};
    uint32_t a{0};
    uint32_t b{0};
    EventType type{EventType::LeftEndpoint};
};

struct SwapCandidate {
    double y{0.0};
    uint32_t lower{0};
    uint32_t upper{0};
};

template <typename T, bool EmitResults> class BentleyOttmannEngine {
public:
    explicit BentleyOttmannEngine(std::span<const LineSegment<T>> raw_segments)
        : segs_(raw_segments.size()), status_(segs_), eq_(raw_segments.size() * 2 + 16) {
        events_.reserve(raw_segments.size() * 2 + 16);

        for (std::size_t i = 0; i < raw_segments.size(); ++i) {
            const double sx = static_cast<double>(raw_segments[i].start[0]);
            const double sy = static_cast<double>(raw_segments[i].start[1]);
            const double ex = static_cast<double>(raw_segments[i].end[0]);
            const double ey = static_cast<double>(raw_segments[i].end[1]);
            if (!std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(ex) ||
                !std::isfinite(ey)) {
                continue;
            }

            Point2D p0{sx, sy};
            Point2D p1{ex, ey};
            if (bo_point_less(p1, p0)) {
                std::swap(p0, p1);
            }

            CanonSegment& cs = segs_[i];
            cs.p1 = p0;
            cs.p2 = p1;
            cs.dx = p1.x - p0.x;
            cs.dy = p1.y - p0.y;
            cs.min_y = std::min(p0.y, p1.y);
            cs.max_y = std::max(p0.y, p1.y);
            cs.orig_index = static_cast<uint32_t>(i);
            cs.is_vertical = (p0.x == p1.x);
            cs.is_point = (p0.x == p1.x && p0.y == p1.y);

            const uint32_t uid = static_cast<uint32_t>(i);
            if (cs.is_vertical) {
                push_event(p0.x, SweepEventRecord{p0.y, uid, uid, EventType::VerticalSegment});
            } else {
                push_event(p0.x, SweepEventRecord{p0.y, uid, uid, EventType::LeftEndpoint});
                push_event(p1.x, SweepEventRecord{p1.y, uid, uid, EventType::RightEndpoint});
            }
        }
    }

    void run() {
        while (!eq_.empty()) {
            const double x0 = eq_.top_key();
            inter_at_x_.clear();
            inter_after_x_.clear();
            r_batch_.clear();
            l_batch_.clear();
            v_batch_.clear();

            while (!eq_.empty() && eq_.top_key() == x0) {
                const auto [ev_id, key] = eq_.pop();
                (void)key;
                const SweepEventRecord& rec = events_[ev_id];
                switch (rec.type) {
                case EventType::IntersectionAtX:
                    inter_at_x_.push_back({rec.y, rec.a, rec.b});
                    break;
                case EventType::RightEndpoint:
                    r_batch_.push_back(rec.a);
                    break;
                case EventType::LeftEndpoint:
                    l_batch_.push_back(rec.a);
                    break;
                case EventType::VerticalSegment:
                    v_batch_.push_back(rec.a);
                    break;
                case EventType::IntersectionAfterX:
                    inter_after_x_.push_back({rec.y, rec.a, rec.b});
                    break;
                }
            }

            // Phase 0: Swap segments crossing at exact x0
            drain_intersections(inter_at_x_, x0, true);

            // Phase 1: Inspect and remove segments ending at x0
            process_right_endpoints(x0);

            // Drain any IntersectionAtX newly formed by right-endpoint deletions
            drain_intersections(inter_at_x_, x0, true);

            // Phase 2: Match R vs L at x0 and insert segments starting at x0
            process_left_endpoints(x0);

            // Phase 3: Process vertical and zero-length point segments at x0
            process_vertical_segments(x0);

            // Phase 4: Swap any remaining intersections local to x0
            drain_intersections(inter_after_x_, x0, false);
        }

        if constexpr (EmitResults) {
            std::sort(results_.begin(), results_.end(),
                      [](const IntersectionResult<T>& lhs, const IntersectionResult<T>& rhs) {
                          if (lhs.segment1_index != rhs.segment1_index) {
                              return lhs.segment1_index < rhs.segment1_index;
                          }
                          return lhs.segment2_index < rhs.segment2_index;
                      });
        }
    }

    [[nodiscard]] std::vector<IntersectionResult<T>> take_results() {
        return std::move(results_);
    }

    [[nodiscard]] std::size_t count() const noexcept {
        return count_;
    }

private:
    void push_event(double x, SweepEventRecord rec) {
        const uint32_t ev_id = static_cast<uint32_t>(events_.size());
        events_.push_back(rec);
        eq_.push(ev_id, x);
    }

    void record_intersection(uint32_t a, uint32_t b, const Point2D& ip) {
        if (reported_pairs_.insert(a, b)) {
            if constexpr (EmitResults) {
                const std::size_t i1 = segs_[a].orig_index;
                const std::size_t i2 = segs_[b].orig_index;
                const std::size_t lo = std::min(i1, i2);
                const std::size_t hi = std::max(i1, i2);
                results_.push_back(
                    IntersectionResult<T>{{static_cast<T>(ip.x), static_cast<T>(ip.y)}, lo, hi});
            } else {
                (void)ip;
                ++count_;
            }
        }
    }

    void test_and_report_pair(uint32_t a, uint32_t b) {
        Point2D ip{};
        if (intersect_segments_exact(segs_[a], segs_[b], ip)) {
            record_intersection(a, b, ip);
        }
    }

    void check_and_schedule(uint32_t lower, uint32_t upper, double x_sweep, bool allow_at_x) {
        if (lower == kBoInvalid || upper == kBoInvalid) {
            return;
        }
        Point2D ip{};
        if (!intersect_segments_exact(segs_[lower], segs_[upper], ip)) {
            return;
        }
        record_intersection(lower, upper, ip);

        if (exact_slope_cmp(segs_[lower], segs_[upper]) > 0) {
            const SwapSchedule sched =
                compute_swap_schedule(segs_[lower], segs_[upper], ip.x, x_sweep);
            if (sched.x <= x_sweep) {
                if (sched.at_x && allow_at_x) {
                    inter_at_x_.push_back({ip.y, lower, upper});
                } else {
                    inter_after_x_.push_back({ip.y, lower, upper});
                }
            } else {
                const EventType t =
                    sched.at_x ? EventType::IntersectionAtX : EventType::IntersectionAfterX;
                push_event(sched.x, SweepEventRecord{ip.y, lower, upper, t});
            }
        }
    }

    void check_outer_band(uint32_t lower, uint32_t upper, double x0, bool allow_at_x) {
        const uint32_t p = status_.prev_of(lower);
        if (p != kBoInvalid) {
            check_and_schedule(p, lower, x0, allow_at_x);
            const double y_lower = eval_y(segs_[lower], x0);
            const double tol = band_tol(segs_[lower], y_lower);
            const Point2D q_lower{x0, y_lower};
            uint32_t pw = status_.prev_of(p);
            while (pw != kBoInvalid &&
                   (orient_seg_point(segs_[pw], q_lower) == VerticalRelation::Coincident ||
                    eval_y(segs_[pw], x0) >= y_lower - tol)) {
                check_and_schedule(pw, lower, x0, allow_at_x);
                pw = status_.prev_of(pw);
            }
        }

        const uint32_t n = status_.next_of(upper);
        if (n != kBoInvalid) {
            check_and_schedule(upper, n, x0, allow_at_x);
            const double y_upper = eval_y(segs_[upper], x0);
            const double tol = band_tol(segs_[upper], y_upper);
            const Point2D q_upper{x0, y_upper};
            uint32_t nw = status_.next_of(n);
            while (nw != kBoInvalid &&
                   (orient_seg_point(segs_[nw], q_upper) == VerticalRelation::Coincident ||
                    eval_y(segs_[nw], x0) <= y_upper + tol)) {
                check_and_schedule(upper, nw, x0, allow_at_x);
                nw = status_.next_of(nw);
            }
        }
    }

    void swap_and_inspect(uint32_t lower, uint32_t upper, double x0, bool allow_at_x) {
        test_and_report_pair(lower, upper);
        status_.swap_adjacent(lower, upper);
        check_outer_band(upper, lower, x0, allow_at_x);
    }

    void drain_intersections(std::vector<SwapCandidate>& queue, double x0, bool allow_at_x) {
        if (queue.size() > 1) {
            // Order descending by y so back() pops in increasing (x, y) order
            std::sort(
                queue.begin(), queue.end(),
                [](const SwapCandidate& lhs, const SwapCandidate& rhs) { return lhs.y > rhs.y; });
        }
        while (!queue.empty()) {
            const SwapCandidate cand = queue.back();
            queue.pop_back();
            const uint32_t a = cand.lower;
            const uint32_t b = cand.upper;
            if (!status_.contains(a) || !status_.contains(b)) {
                continue;
            }
            if (status_.next_of(a) == b) {
                swap_and_inspect(a, b, x0, allow_at_x);
                continue;
            }

            const double ya = eval_y(segs_[a], x0);
            const double tol = band_tol(segs_[a], ya);
            uint32_t cur = status_.next_of(a);
            bool can_bubble = false;
            for (int step = 0; step < 4 && cur != kBoInvalid; ++step) {
                if (std::abs(eval_y(segs_[cur], x0) - ya) > tol ||
                    exact_slope_cmp(segs_[a], segs_[cur]) <= 0) {
                    break;
                }
                if (cur == b) {
                    can_bubble = true;
                    break;
                }
                cur = status_.next_of(cur);
            }
            if (!can_bubble) {
                continue;
            }
            while (status_.next_of(a) != b) {
                swap_and_inspect(a, status_.next_of(a), x0, allow_at_x);
            }
            swap_and_inspect(a, b, x0, allow_at_x);
        }
    }

    void inspect_coincident_neighbors(uint32_t u, double x0, double y_ref, bool allow_at_x) {
        const Point2D q_ref{x0, y_ref};
        uint32_t p = status_.prev_of(u);
        if (p != kBoInvalid) {
            check_and_schedule(p, u, x0, allow_at_x);
            const double tol = band_tol(segs_[u], y_ref);
            while (true) {
                p = status_.prev_of(p);
                if (p == kBoInvalid ||
                    (orient_seg_point(segs_[p], q_ref) != VerticalRelation::Coincident &&
                     eval_y(segs_[p], x0) < y_ref - tol)) {
                    break;
                }
                check_and_schedule(p, u, x0, allow_at_x);
            }
        }

        uint32_t n = status_.next_of(u);
        if (n != kBoInvalid) {
            check_and_schedule(u, n, x0, allow_at_x);
            const double tol = band_tol(segs_[u], y_ref);
            while (true) {
                n = status_.next_of(n);
                if (n == kBoInvalid ||
                    (orient_seg_point(segs_[n], q_ref) != VerticalRelation::Coincident &&
                     eval_y(segs_[n], x0) > y_ref + tol)) {
                    break;
                }
                check_and_schedule(u, n, x0, allow_at_x);
            }
        }
    }

    void process_right_endpoints(double x0) {
        if (r_batch_.empty()) {
            return;
        }
        if (r_batch_.size() > 1) {
            std::sort(r_batch_.begin(), r_batch_.end(),
                      [&](uint32_t a, uint32_t b) { return segs_[a].p2.y < segs_[b].p2.y; });
            for (std::size_t i = 0; i < r_batch_.size(); ++i) {
                for (std::size_t j = i + 1;
                     j < r_batch_.size() && segs_[r_batch_[j]].p2.y == segs_[r_batch_[i]].p2.y;
                     ++j) {
                    test_and_report_pair(r_batch_[i], r_batch_[j]);
                }
            }
        }

        for (uint32_t r : r_batch_) {
            if (!status_.contains(r)) {
                continue;
            }
            const double yr = segs_[r].p2.y;
            while (true) {
                const uint32_t p = status_.prev_of(r);
                if (p == kBoInvalid || eval_y(segs_[p], x0) <= yr) {
                    break;
                }
                test_and_report_pair(p, r);
                status_.swap_adjacent(p, r);
            }
            while (true) {
                const uint32_t n = status_.next_of(r);
                if (n == kBoInvalid || eval_y(segs_[n], x0) >= yr) {
                    break;
                }
                test_and_report_pair(r, n);
                status_.swap_adjacent(r, n);
            }
            inspect_coincident_neighbors(r, x0, yr, true);
        }

        surviving_.clear();
        for (uint32_t r : r_batch_) {
            if (!status_.contains(r)) {
                continue;
            }
            const uint32_t p = status_.prev_of(r);
            const uint32_t n = status_.next_of(r);
            status_.erase(r);
            if (p != kBoInvalid) {
                surviving_.push_back(p);
            }
            if (n != kBoInvalid) {
                surviving_.push_back(n);
            }
        }

        for (uint32_t s : surviving_) {
            if (!status_.contains(s)) {
                continue;
            }
            const uint32_t p = status_.prev_of(s);
            const uint32_t n = status_.next_of(s);
            if (p != kBoInvalid) {
                check_and_schedule(p, s, x0, true);
            }
            if (n != kBoInvalid) {
                check_and_schedule(s, n, x0, true);
            }
        }
    }

    void process_left_endpoints(double x0) {
        if (l_batch_.empty()) {
            return;
        }
        if (l_batch_.size() > 1) {
            std::sort(l_batch_.begin(), l_batch_.end(), [&](uint32_t a, uint32_t b) {
                if (segs_[a].p1.y != segs_[b].p1.y) {
                    return segs_[a].p1.y < segs_[b].p1.y;
                }
                const int sc = exact_slope_cmp(segs_[a], segs_[b]);
                if (sc != 0) {
                    return sc < 0;
                }
                return a < b;
            });
        }

        if (!r_batch_.empty()) {
            std::size_t ri = 0;
            std::size_t li = 0;
            while (ri < r_batch_.size() && li < l_batch_.size()) {
                const double ry = segs_[r_batch_[ri]].p2.y;
                const double ly = segs_[l_batch_[li]].p1.y;
                if (ry < ly) {
                    ++ri;
                } else if (ly < ry) {
                    ++li;
                } else {
                    std::size_t r_end = ri + 1;
                    while (r_end < r_batch_.size() && segs_[r_batch_[r_end]].p2.y == ry) {
                        ++r_end;
                    }
                    std::size_t l_end = li + 1;
                    while (l_end < l_batch_.size() && segs_[l_batch_[l_end]].p1.y == ly) {
                        ++l_end;
                    }
                    for (std::size_t a = ri; a < r_end; ++a) {
                        for (std::size_t b = li; b < l_end; ++b) {
                            test_and_report_pair(r_batch_[a], l_batch_[b]);
                        }
                    }
                    ri = r_end;
                    li = l_end;
                }
            }
        }

        for (uint32_t l : l_batch_) {
            status_.insert(l, x0);
            inspect_coincident_neighbors(l, x0, segs_[l].p1.y, false);
        }
    }

    void process_vertical_segments(double x0) {
        if (v_batch_.empty()) {
            return;
        }
        if (v_batch_.size() > 1) {
            std::sort(v_batch_.begin(), v_batch_.end(), [&](uint32_t a, uint32_t b) {
                if (segs_[a].p1.y != segs_[b].p1.y) {
                    return segs_[a].p1.y < segs_[b].p1.y;
                }
                if (segs_[a].p2.y != segs_[b].p2.y) {
                    return segs_[a].p2.y < segs_[b].p2.y;
                }
                return a < b;
            });

            active_verts_.clear();
            for (uint32_t v : v_batch_) {
                const double vy1 = segs_[v].p1.y;
                std::size_t write_idx = 0;
                for (std::size_t j = 0; j < active_verts_.size(); ++j) {
                    const uint32_t w = active_verts_[j];
                    if (segs_[w].p2.y >= vy1) {
                        test_and_report_pair(w, v);
                        active_verts_[write_idx++] = w;
                    }
                }
                active_verts_.resize(write_idx);
                active_verts_.push_back(v);
            }
        }

        for (uint32_t v : v_batch_) {
            const double vy1 = segs_[v].p1.y;
            const double vy2 = segs_[v].p2.y;
            const Point2D q_top{x0, vy2};

            if (!r_batch_.empty()) {
                auto it = std::lower_bound(
                    r_batch_.begin(), r_batch_.end(), vy1,
                    [&](uint32_t r, double y_val) { return segs_[r].p2.y < y_val; });
                while (it != r_batch_.end() && segs_[*it].p2.y <= vy2) {
                    test_and_report_pair(*it, v);
                    ++it;
                }
            }

            uint32_t cur = status_.find_vertical_lower_bound(x0, vy1);
            while (cur != kBoInvalid &&
                   (orient_seg_point(segs_[cur], q_top) != VerticalRelation::Above ||
                    eval_y(segs_[cur], x0) <= vy2 + band_tol(segs_[cur], vy2))) {
                test_and_report_pair(cur, v);
                cur = status_.next_of(cur);
            }
        }
    }

    std::vector<CanonSegment> segs_;
    StatusTreap status_;
    std::vector<SweepEventRecord> events_;
    algoat::graph::DAryHeap<4, double, uint32_t> eq_;

    std::vector<SwapCandidate> inter_at_x_;
    std::vector<SwapCandidate> inter_after_x_;
    std::vector<uint32_t> r_batch_;
    std::vector<uint32_t> l_batch_;
    std::vector<uint32_t> v_batch_;
    std::vector<uint32_t> surviving_;
    std::vector<uint32_t> active_verts_;

    PairSet reported_pairs_;
    std::vector<IntersectionResult<T>> results_;
    std::size_t count_{0};
};

} // namespace detail

/**
 * @brief Computes all pairwise intersection points among @p segments using Bentley-Ottmann sweep.
 *
 * Returns intersections sorted deterministically by @c (segment1_index, segment2_index) with
 * @c segment1_index < @c segment2_index, matching brute-force O(N^2) enumeration.
 */
template <typename T = double>
[[nodiscard]] std::vector<IntersectionResult<T>>
bentley_ottmann(std::span<const LineSegment<T>> segments) {
    if (segments.size() < 2) {
        return {};
    }
    detail::BentleyOttmannEngine<T, true> engine(segments);
    engine.run();
    return engine.take_results();
}

template <typename T>
[[nodiscard]] std::vector<IntersectionResult<T>>
bentley_ottmann(std::span<LineSegment<T>> segments) {
    return bentley_ottmann(std::span<const LineSegment<T>>{segments.data(), segments.size()});
}

template <std::ranges::contiguous_range R>
    requires detail::IsLineSegmentRange<R>
[[nodiscard]] auto bentley_ottmann(R&& segments) {
    using SegType = std::remove_cvref_t<std::ranges::range_value_t<R>>;
    using T = typename SegType::value_type;
    return bentley_ottmann(
        std::span<const LineSegment<T>>{std::ranges::data(segments), std::ranges::size(segments)});
}

/**
 * @brief Counts total intersecting segment pairs without allocating result structs.
 */
template <typename T = double>
[[nodiscard]] std::size_t count_intersections(std::span<const LineSegment<T>> segments) {
    if (segments.size() < 2) {
        return 0;
    }
    detail::BentleyOttmannEngine<T, false> engine(segments);
    engine.run();
    return engine.count();
}

template <typename T>
[[nodiscard]] std::size_t count_intersections(std::span<LineSegment<T>> segments) {
    return count_intersections(std::span<const LineSegment<T>>{segments.data(), segments.size()});
}

template <std::ranges::contiguous_range R>
    requires detail::IsLineSegmentRange<R>
[[nodiscard]] std::size_t count_intersections(R&& segments) {
    using SegType = std::remove_cvref_t<std::ranges::range_value_t<R>>;
    using T = typename SegType::value_type;
    return count_intersections(
        std::span<const LineSegment<T>>{std::ranges::data(segments), std::ranges::size(segments)});
}

} // namespace algoat::geometry
