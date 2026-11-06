/**
 * @file optimal_pla.hpp
 * @brief Streaming Optimal Piecewise Linear Approximation (OPLA) based on O'Rourke (1981).
 *
 * Implements an amortized O(1) time per point streaming algorithm that partitions a sorted
 * sequence of keys into the minimum number of linear segments bounded by maximum error epsilon.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::searching::detail {

template <typename T>
using LargeSigned = std::conditional_t<std::is_floating_point_v<T>, long double,
                                       std::conditional_t<(sizeof(T) < 8), int64_t,
#if defined(__SIZEOF_INT128__)
                                                          __int128
#else
                                                          int64_t
#endif
                                                          >>;

/**
 * @class OptimalPiecewiseLinearModel
 * @brief Computes the optimal piecewise linear segmentation under L-infinity norm error epsilon.
 *
 * Employs Joseph O'Rourke's (1981) streaming cone intersection algorithm, maintaining the
 * upper and lower convex hulls of tolerance intervals to determine feasible line slopes in
 * linear time.
 */
template <typename X, typename Y = std::size_t> class OptimalPiecewiseLinearModel {
private:
    using SX = LargeSigned<X>;
    using SY = LargeSigned<Y>;

    struct Slope {
        SX dx{0};
        SY dy{0};

        constexpr bool operator<(const Slope& p) const noexcept {
            return dy * p.dx < dx * p.dy;
        }
        constexpr bool operator>(const Slope& p) const noexcept {
            return dy * p.dx > dx * p.dy;
        }
        constexpr bool operator==(const Slope& p) const noexcept {
            return dy * p.dx == dx * p.dy;
        }
        constexpr bool operator!=(const Slope& p) const noexcept {
            return dy * p.dx != dx * p.dy;
        }

        [[nodiscard]] explicit operator long double() const noexcept {
            return static_cast<long double>(dy) / static_cast<long double>(dx);
        }
    };

    struct Point {
        X x{};
        Y y{};

        constexpr Slope operator-(const Point& p) const noexcept {
            return Slope{static_cast<SX>(x) - static_cast<SX>(p.x),
                         static_cast<SY>(y) - static_cast<SY>(p.y)};
        }
    };

    const Y epsilon_;
    std::vector<Point> lower_;
    std::vector<Point> upper_;
    X first_x_{0};
    X last_x_{0};
    std::size_t lower_start_{0};
    std::size_t upper_start_{0};
    std::size_t points_in_hull_{0};
    Point rectangle_[4]{};

    static constexpr auto cross(const Point& O, const Point& A, const Point& B) noexcept {
        auto OA = A - O;
        auto OB = B - O;
        return OA.dx * OB.dy - OA.dy * OB.dx;
    }

public:
    class CanonicalSegment {
        friend class OptimalPiecewiseLinearModel;
        Point rectangle_[4]{};
        X first_{};

        CanonicalSegment(const Point& p0, const Point& p1, X first)
            : rectangle_{p0, p1, p0, p1}, first_(first) {}

        CanonicalSegment(const Point (&rect)[4], X first)
            : rectangle_{rect[0], rect[1], rect[2], rect[3]}, first_(first) {}

        [[nodiscard]] bool one_point() const noexcept {
            return rectangle_[0].x == rectangle_[2].x && rectangle_[0].y == rectangle_[2].y &&
                   rectangle_[1].x == rectangle_[3].x && rectangle_[1].y == rectangle_[3].y;
        }

    public:
        CanonicalSegment() = default;

        [[nodiscard]] X get_first_x() const noexcept {
            return first_;
        }

        [[nodiscard]] std::pair<float, int32_t>
        get_floating_point_segment(const X& origin) const noexcept {
            if (one_point()) {
                return {0.0f, static_cast<int32_t>((rectangle_[0].y + rectangle_[1].y) / 2)};
            }

            if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) {
                auto slope = rectangle_[3] - rectangle_[1];
                auto intercept_n = slope.dy * (static_cast<SX>(origin) - rectangle_[1].x);
                auto intercept_d = slope.dx;
                auto rounding_term =
                    ((intercept_n < 0) ^ (intercept_d < 0) ? -1 : +1) * intercept_d / 2;
                auto intercept = (intercept_n + rounding_term) / intercept_d + rectangle_[1].y;
                return {static_cast<float>(static_cast<long double>(slope)),
                        static_cast<int32_t>(intercept)};
            } else {
                auto slope1 = rectangle_[2] - rectangle_[0];
                auto slope2 = rectangle_[3] - rectangle_[1];
                if (slope1 == slope2) {
                    return {static_cast<float>(static_cast<long double>(slope1)),
                            static_cast<int32_t>(rectangle_[0].y)};
                }

                auto p0p1 = rectangle_[1] - rectangle_[0];
                auto a = slope1.dx * slope2.dy - slope1.dy * slope2.dx;
                auto b = (p0p1.dx * slope2.dy - p0p1.dy * slope2.dx) / static_cast<long double>(a);
                auto i_x = rectangle_[0].x + b * slope1.dx;
                auto i_y = rectangle_[0].y + b * slope1.dy;

                auto min_slope = static_cast<long double>(slope1);
                auto max_slope = static_cast<long double>(slope2);
                auto slope = (min_slope + max_slope) / 2.0;
                auto intercept = i_y - (i_x - origin) * slope;
                return {static_cast<float>(slope), static_cast<int32_t>(std::round(intercept))};
            }
        }

        [[nodiscard]] std::pair<float, int32_t> get_floating_point_segment() const noexcept {
            return get_floating_point_segment(first_);
        }
    };

    explicit OptimalPiecewiseLinearModel(Y epsilon) : epsilon_(epsilon) {
        upper_.reserve(1024);
        lower_.reserve(1024);
    }

    bool add_point(const X& x, const Y& y) {
        if (points_in_hull_ > 0 && x <= last_x_) {
            return false;
        }

        last_x_ = x;
        auto max_y = std::numeric_limits<Y>::max();
        auto min_y = std::numeric_limits<Y>::lowest();
        Point p1{x, y >= max_y - epsilon_ ? max_y : y + epsilon_};
        Point p2{x, y <= min_y + epsilon_ ? min_y : y - epsilon_};

        if (points_in_hull_ == 0) {
            first_x_ = x;
            rectangle_[0] = p1;
            rectangle_[1] = p2;
            upper_.clear();
            lower_.clear();
            upper_.push_back(p1);
            lower_.push_back(p2);
            upper_start_ = lower_start_ = 0;
            ++points_in_hull_;
            return true;
        }

        if (points_in_hull_ == 1) {
            rectangle_[2] = p2;
            rectangle_[3] = p1;
            upper_.push_back(p1);
            lower_.push_back(p2);
            ++points_in_hull_;
            return true;
        }

        auto slope1 = rectangle_[2] - rectangle_[0];
        auto slope2 = rectangle_[3] - rectangle_[1];
        bool outside_line1 = (p1 - rectangle_[2] < slope1);
        bool outside_line2 = (p2 - rectangle_[3] > slope2);

        if (outside_line1 || outside_line2) {
            points_in_hull_ = 0;
            return false;
        }

        // Restrict maximum slope
        if (p1 - rectangle_[1] < slope2) {
            auto min_val = lower_[lower_start_] - p1;
            auto min_i = lower_start_;
            for (auto i = lower_start_ + 1; i < lower_.size(); ++i) {
                auto val = lower_[i] - p1;
                if (val > min_val) {
                    break;
                }
                min_val = val;
                min_i = i;
            }
            rectangle_[1] = lower_[min_i];
            rectangle_[3] = p1;
            lower_start_ = min_i;

            auto end = upper_.size();
            for (; end >= upper_start_ + 2 && cross(upper_[end - 2], upper_[end - 1], p1) <= 0;
                 --end) {
                continue;
            }
            upper_.resize(end);
            upper_.push_back(p1);
        }

        // Restrict minimum slope
        if (p2 - rectangle_[0] > slope1) {
            auto max_val = upper_[upper_start_] - p2;
            auto max_i = upper_start_;
            for (auto i = upper_start_ + 1; i < upper_.size(); ++i) {
                auto val = upper_[i] - p2;
                if (val < max_val) {
                    break;
                }
                max_val = val;
                max_i = i;
            }
            rectangle_[0] = upper_[max_i];
            rectangle_[2] = p2;
            upper_start_ = max_i;

            auto end = lower_.size();
            for (; end >= lower_start_ + 2 && cross(lower_[end - 2], lower_[end - 1], p2) >= 0;
                 --end) {
                continue;
            }
            lower_.resize(end);
            lower_.push_back(p2);
        }

        ++points_in_hull_;
        return true;
    }

    [[nodiscard]] CanonicalSegment get_segment() const noexcept {
        if (points_in_hull_ == 1) {
            return CanonicalSegment(rectangle_[0], rectangle_[1], first_x_);
        }
        return CanonicalSegment(rectangle_, first_x_);
    }

    void reset() noexcept {
        points_in_hull_ = 0;
        lower_.clear();
        upper_.clear();
    }
};

/**
 * @brief Streams elements and constructs optimal piecewise linear segments.
 *
 * @tparam KeyAccessor Callable returning key value for index i.
 * @tparam SegmentConsumer Callable accepting constructed CanonicalSegment.
 * @param n Number of data elements.
 * @param epsilon Maximum allowable absolute error.
 * @param key_at Function accessing key at index i.
 * @param consume_segment Consumer receiving finalized canonical segments.
 * @return Total number of segments produced.
 */
template <typename KeyAccessor, typename SegmentConsumer>
std::size_t make_segmentation(std::size_t n, std::size_t epsilon, KeyAccessor key_at,
                              SegmentConsumer consume_segment) {
    if (n == 0) {
        return 0;
    }

    using K = std::invoke_result_t<KeyAccessor, std::size_t>;
    std::size_t segments_count = 0;
    OptimalPiecewiseLinearModel<K, std::size_t> opt(epsilon);

    auto add_point = [&](K x, std::size_t y) {
        if (!opt.add_point(x, y)) {
            consume_segment(opt.get_segment());
            opt.add_point(x, y);
            ++segments_count;
        }
    };

    add_point(key_at(0), 0);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (key_at(i) == key_at(i - 1)) {
            if constexpr (std::is_integral_v<K>) {
                if (key_at(i) + 1 < key_at(i + 1)) {
                    add_point(key_at(i) + 1, i);
                }
            }
        } else {
            add_point(key_at(i), i);
        }
    }

    if (n >= 2 && key_at(n - 1) != key_at(n - 2)) {
        add_point(key_at(n - 1), n - 1);
    }

    // Anchor tail point for queries beyond the maximum key
    if constexpr (std::is_integral_v<K>) {
        if (key_at(n - 1) < std::numeric_limits<K>::max()) {
            add_point(key_at(n - 1) + 1, n);
        }
    }

    consume_segment(opt.get_segment());
    return ++segments_count;
}

} // namespace algoat::searching::detail
