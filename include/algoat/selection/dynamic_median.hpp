/**
 * @file dynamic_median.hpp
 * @brief Dynamic stream median and sliding window order statistics engines.
 */

#pragma once

#include "algoat/selection/detail/d_ary_heap.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::selection {

namespace detail {

template <typename Comp> struct InvertComp {
    [[no_unique_address]] Comp comp;

    template <typename T, typename U>
    constexpr bool operator()(const T& a, const U& b) const noexcept(noexcept(comp(b, a))) {
        return comp(b, a);
    }
};

template <typename Multiset, typename T>
inline const T& select_rank_multiset(const Multiset& ms, std::size_t rank) {
    if (rank <= ms.size() / 2) {
        return *std::next(ms.begin(), static_cast<std::ptrdiff_t>(rank));
    }
    const auto dist = ms.size() - 1 - rank;
    return *std::next(ms.rbegin(), static_cast<std::ptrdiff_t>(dist));
}

} // namespace detail

/**
 * @class DynamicMedian
 * @brief Online unbounded dynamic stream median using two balanced 4-ary heaps.
 *
 * Maintains a lower max-heap and upper min-heap. Insertion runs in O(log N) time,
 * and median query runs in strictly O(1) time.
 *
 * @tparam T Element type.
 * @tparam Comp Strict weak ordering comparator (default std::ranges::less).
 */
template <typename T, typename Comp = std::ranges::less> class DynamicMedian {
public:
    using value_type = T;
    using size_type = std::size_t;
    using compare_type = Comp;

    DynamicMedian() = default;

    explicit DynamicMedian(Comp comp)
        : comp_(comp), lower_half_(comp), upper_half_(detail::InvertComp<Comp>{comp}) {}

    /**
     * @brief Adds a new streaming element in O(log N).
     */
    void push(const T& value) {
        if (lower_half_.empty() || !comp_(lower_half_.top(), value)) {
            lower_half_.push(value);
        } else {
            upper_half_.push(value);
        }

        rebalance();
    }

    void push(T&& value) {
        if (lower_half_.empty() || !comp_(lower_half_.top(), value)) {
            lower_half_.push(std::move(value));
        } else {
            upper_half_.push(std::move(value));
        }

        rebalance();
    }

    /**
     * @brief Queries the current median in O(1).
     * @throws std::underflow_error if empty().
     */
    [[nodiscard]] double median() const {
        if (empty()) {
            throw std::underflow_error("median() called on empty DynamicMedian");
        }
        if (lower_half_.size() > upper_half_.size()) {
            return static_cast<double>(lower_half_.top());
        }
        if constexpr (std::is_arithmetic_v<T>) {
            return static_cast<double>(std::midpoint(static_cast<double>(lower_half_.top()),
                                                     static_cast<double>(upper_half_.top())));
        } else {
            return (static_cast<double>(lower_half_.top()) +
                    static_cast<double>(upper_half_.top())) *
                   0.5;
        }
    }

    [[nodiscard]] size_type size() const noexcept {
        return lower_half_.size() + upper_half_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return lower_half_.empty() && upper_half_.empty();
    }

    void clear() noexcept {
        lower_half_.clear();
        upper_half_.clear();
    }

    void reserve(size_type capacity) {
        lower_half_.reserve((capacity + 1) / 2);
        upper_half_.reserve(capacity / 2);
    }

private:
    [[no_unique_address]] Comp comp_{};

    // 4-ary max-heap (lower half)
    detail::d_ary_heap<T, 4, Comp> lower_half_{comp_};

    // 4-ary min-heap (upper half)
    detail::d_ary_heap<T, 4, detail::InvertComp<Comp>> upper_half_{detail::InvertComp<Comp>{comp_}};

    void rebalance() {
        if (lower_half_.size() > upper_half_.size() + 1) {
            upper_half_.push(lower_half_.top());
            lower_half_.pop();
        } else if (lower_half_.size() < upper_half_.size()) {
            lower_half_.push(upper_half_.top());
            upper_half_.pop();
        }
    }
};

/**
 * @class SlidingWindowMedian
 * @brief Rolling window median and order statistics over a fixed window size W.
 *
 * Uses a circular ring buffer for O(1) eviction tracking and a balanced dual multiset
 * with C++17 node handle transfer for zero-allocation rebalancing and steady-state updates.
 *
 * @tparam T Element type.
 * @tparam Comp Strict weak ordering comparator (default std::ranges::less).
 */
template <typename T, typename Comp = std::ranges::less> class SlidingWindowMedian {
public:
    using value_type = T;
    using size_type = std::size_t;
    using compare_type = Comp;

    explicit SlidingWindowMedian(size_type window_size, Comp comp = Comp{})
        : window_size_{window_size}, comp_{std::move(comp)}, lower_window_{comp_},
          upper_window_{comp_} {
        if (window_size_ == 0) {
            throw std::invalid_argument("window_size must be greater than 0");
        }
        ring_buffer_.reserve(window_size_);
    }

    /**
     * @brief Slides the window by 1: removes oldest element, inserts incoming value in O(log W).
     * Steady-state sliding achieves zero heap allocations by recycling extracted node handles.
     */
    void push(const T& value) {
        push_impl(value);
    }

    void push(T&& value) {
        push_impl(std::move(value));
    }

    /**
     * @brief Returns current median in O(1).
     * @throws std::underflow_error if empty().
     */
    [[nodiscard]] double median() const {
        if (empty()) {
            throw std::underflow_error("median() called on empty SlidingWindowMedian");
        }
        if (count_ % 2 != 0) {
            return static_cast<double>(*lower_window_.rbegin());
        }
        if constexpr (std::is_arithmetic_v<T>) {
            return static_cast<double>(std::midpoint(static_cast<double>(*lower_window_.rbegin()),
                                                     static_cast<double>(*upper_window_.begin())));
        } else {
            return (static_cast<double>(*lower_window_.rbegin()) +
                    static_cast<double>(*upper_window_.begin())) *
                   0.5;
        }
    }

    /**
     * @brief Queries arbitrary k-th order statistic within the current window in O(log W)
     * amortized.
     * @param rank Zero-based rank (0 <= rank < size()).
     * @throws std::out_of_range if rank >= size().
     */
    [[nodiscard]] const T& order_statistic(size_type rank) const {
        if (rank >= count_) {
            throw std::out_of_range("order_statistic rank out of bounds");
        }

        if (rank < lower_window_.size()) {
            return detail::select_rank_multiset<std::multiset<T, Comp>, T>(lower_window_, rank);
        }
        return detail::select_rank_multiset<std::multiset<T, Comp>, T>(upper_window_,
                                                                       rank - lower_window_.size());
    }

    [[nodiscard]] bool full() const noexcept {
        return count_ == window_size_;
    }

    [[nodiscard]] size_type size() const noexcept {
        return count_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0;
    }

    [[nodiscard]] size_type window_size() const noexcept {
        return window_size_;
    }

    void clear() noexcept {
        head_ = 0;
        count_ = 0;
        ring_buffer_.clear();
        lower_window_.clear();
        upper_window_.clear();
    }

private:
    size_type window_size_{0};
    size_type count_{0};
    size_type head_{0};
    [[no_unique_address]] Comp comp_{};

    // Circular queue tracking arrival order for O(1) eviction
    std::vector<T> ring_buffer_;

    // Internal balanced dual-multiset with node handle transfer
    std::multiset<T, Comp> lower_window_;
    std::multiset<T, Comp> upper_window_;

    template <typename U> void push_impl(U&& value) {
        if (full()) {
            // Locate oldest element and extract node handle for in-place recycling (zero
            // allocation)
            const T& oldest = ring_buffer_[head_];
            auto nh = extract_oldest_node(oldest);

            // Overwrite node value in-place without reallocation
            nh.value() = std::forward<U>(value);
            ring_buffer_[head_] = nh.value();
            head_ = (head_ + 1 == window_size_) ? 0 : head_ + 1;

            // Route recycled node handle into lower or upper window
            if (!upper_window_.empty() && comp_(*upper_window_.begin(), nh.value())) {
                upper_window_.insert(std::move(nh));
            } else {
                lower_window_.insert(std::move(nh));
            }
        } else {
            ring_buffer_.push_back(value);
            ++count_;

            // Initial warm-up routing
            if (!upper_window_.empty() && comp_(*upper_window_.begin(), value)) {
                upper_window_.insert(std::forward<U>(value));
            } else {
                lower_window_.insert(std::forward<U>(value));
            }
        }

        rebalance();
    }

    typename std::multiset<T, Comp>::node_type extract_oldest_node(const T& oldest) {
        if (!lower_window_.empty() && !comp_(*lower_window_.rbegin(), oldest)) {
            auto it = lower_window_.find(oldest);
            if (it != lower_window_.end()) {
                return lower_window_.extract(it);
            }
        }
        auto uit = upper_window_.find(oldest);
        if (uit != upper_window_.end()) {
            return upper_window_.extract(uit);
        }
        auto it = lower_window_.find(oldest);
        return lower_window_.extract(it);
    }

    void rebalance() {
        const size_type target_lower = (count_ + 1) / 2;
        while (lower_window_.size() > target_lower) {
            auto it = std::prev(lower_window_.end());
            auto nh = lower_window_.extract(it);
            upper_window_.insert(std::move(nh));
        }
        while (lower_window_.size() < target_lower && !upper_window_.empty()) {
            auto it = upper_window_.begin();
            auto nh = upper_window_.extract(it);
            lower_window_.insert(std::move(nh));
        }

        // Cross-boundary invariant guarantee
        while (!lower_window_.empty() && !upper_window_.empty() &&
               comp_(*upper_window_.begin(), *lower_window_.rbegin())) {
            auto l_it = std::prev(lower_window_.end());
            auto u_it = upper_window_.begin();
            auto l_node = lower_window_.extract(l_it);
            auto u_node = upper_window_.extract(u_it);
            lower_window_.insert(std::move(u_node));
            upper_window_.insert(std::move(l_node));
        }
    }
};

/**
 * @brief Computes sliding window median over an entire range.
 *
 * @tparam R Input range type.
 * @tparam Out Output iterator type.
 * @tparam Comp Strict weak ordering comparator.
 * @param data Range of data values.
 * @param window_size Sliding window length W.
 * @param out Output iterator receiving median values.
 * @param comp Optional custom comparator.
 * @return Advanced output iterator.
 */
template <std::ranges::input_range R, std::weakly_incrementable Out,
          typename Comp = std::ranges::less>
Out sliding_window_median(R&& data, std::size_t window_size, Out out, Comp comp = Comp{}) {
    if (window_size == 0) {
        return out;
    }

    using ValueType = std::ranges::range_value_t<R>;
    SlidingWindowMedian<ValueType, Comp> window(window_size, comp);

    auto it = std::ranges::begin(data);
    const auto last = std::ranges::end(data);

    // Warm-up initial window [0, W - 1)
    for (std::size_t i = 1; i < window_size && it != last; ++i, ++it) {
        window.push(*it);
    }

    // Stream and output medians
    for (; it != last; ++it) {
        window.push(*it);
        *out = window.median();
        ++out;
    }

    return out;
}

} // namespace algoat::selection
