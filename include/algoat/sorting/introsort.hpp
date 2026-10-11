/**
 * @file introsort.hpp
 * @brief Introspective Sort (IntroSort) hybrid algorithm implementation.
 */

#pragma once

#include "algoat/core/traits.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail {

/**
 * @brief Returns the minimum of two values according to the comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param a First value.
 * @param b Second value.
 * @param comp Strict weak ordering comparator.
 * @return Smaller of a or b.
 */
template <typename T, typename Compare> constexpr T min_val(const T& a, const T& b, Compare comp) {
    return comp(b, a) ? b : a;
}

/**
 * @brief Returns the maximum of two values according to the comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param a First value.
 * @param b Second value.
 * @param comp Strict weak ordering comparator.
 * @return Larger of a or b.
 */
template <typename T, typename Compare> constexpr T max_val(const T& a, const T& b, Compare comp) {
    return comp(a, b) ? b : a;
}

/**
 * @brief Computes the median of three values according to the comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param a First value.
 * @param b Second value.
 * @param c Third value.
 * @param comp Strict weak ordering comparator.
 * @return Median of a, b, and c.
 */
template <typename T, typename Compare>
constexpr T get_pivot(const T& a, const T& b, const T& c, Compare comp) {
    return max_val(min_val(a, b, comp), min_val(max_val(a, b, comp), c, comp), comp);
}

/**
 * @brief Bounds-safe ISO C++20 Hoare partitioning algorithm.
 *
 * Guarantees all iterator increments and decrements remain strictly within
 * the valid object range [first, last], eliminating undefined pointer arithmetic
 * (ISO/IEC 14882:2020 §7.6.6 [expr.add]).
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 * @param first Beginning of the range to partition.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 * @return Iterator pointing to the partition cut point strictly in (first, last).
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr I hoare_partition(I first, I last, Comp comp, Proj proj) {
    auto len = last - first;
    auto mid = first + len / 2;

    if (std::invoke(comp, std::invoke(proj, *mid), std::invoke(proj, *first))) {
        std::ranges::iter_swap(first, mid);
    }
    if (std::invoke(comp, std::invoke(proj, *(last - 1)), std::invoke(proj, *first))) {
        std::ranges::iter_swap(first, last - 1);
    }
    if (std::invoke(comp, std::invoke(proj, *(last - 1)), std::invoke(proj, *mid))) {
        std::ranges::iter_swap(mid, last - 1);
    }
    std::ranges::iter_swap(first, mid);

    auto pivot = std::invoke(proj, *first);
    I left = first;
    I right = last;

    while (true) {
        while (left < last && std::invoke(comp, std::invoke(proj, *left), pivot)) {
            ++left;
        }
        do {
            --right;
        } while (right > first && std::invoke(comp, pivot, std::invoke(proj, *right)));
        if (left >= right) {
            return right + 1;
        }
        std::ranges::iter_swap(left, right);
        ++left;
    }
}

/**
 * @brief Recursive introsort driver loop bounding recursion depth.
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 * @param first Beginning of the subrange.
 * @param last Sentinel or end of the subrange.
 * @param depth_limit Remaining recursion depth before fallback to heapsort.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void introsort_loop(I first, I last, int depth_limit, Comp comp, Proj proj) {
    while (last - first > 32) {
        if (depth_limit == 0) {
            heapsort(first, last, comp, proj);
            return;
        }
        --depth_limit;
        I cut = hoare_partition(first, last, comp, proj);
        introsort_loop(cut, last, depth_limit, comp, proj);
        last = cut;
    }
    insertion_sort(first, last, comp, proj);
}

/**
 * @brief Implementation bridge for span-based introsort.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param data Contiguous span of elements.
 * @param depth_limit Recursion depth limit.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare>
constexpr void introsort_impl(std::span<T> data, int depth_limit, Compare comp) {
    if (data.empty()) {
        return;
    }
    introsort_loop(data.begin(), data.end(), depth_limit, comp, std::identity{});
}

} // namespace detail

/**
 * @brief Sorts a range delimited by iterators [first, last) in-place using introsort with custom
 * comparator and projection.
 *
 * @tparam I Random-access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range to sort in-place.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr void introsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    auto len = last_it - first;
    if (len <= 1) {
        return;
    }
    int depth_limit = 2 * std::bit_width(static_cast<std::size_t>(len));
    detail::introsort_loop(first, last_it, depth_limit, comp, proj);
}

/**
 * @brief Sorts a C++20 random-access range in-place using introsort with custom comparator and
 * projection.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param data Range of elements to sort in-place.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> introsort(R&& data, Comp comp = {}, Proj proj = {}) {
    introsort(std::ranges::begin(data), std::ranges::end(data), comp, proj);
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

/**
 * @brief Sorts the span in-place using introsort with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> constexpr void introsort(std::span<T> data, Compare comp) {
    if (data.empty())
        return;
    int depth_limit = 2 * std::bit_width(data.size());
    detail::introsort_impl(data, depth_limit, comp);
}

/**
 * @brief Sorts the span in-place using introsort.
 *
 * @tparam T Element type supporting @c operator<.
 * @param data Contiguous span of elements to sort.
 */
template <typename T> constexpr void introsort(std::span<T> data) {
    introsort(data, std::less<T>{});
}

/**
 * @struct IntroSort
 * @brief Hybrid sorting algorithm combining QuickSort, HeapSort, and InsertionSort.
 *
 * Developed by David Musser (1997), IntroSort begins with QuickSort and monitors
 * recursion depth. If the recursion depth exceeds @c 2 * floor(log2(N)), it
 * switches to HeapSort to prevent quadratic worst-case degradation. Small partitions
 * (@c <= 32 elements) are sorted using InsertionSort.
 *
 * @par Characteristics:
 * - <b>Category:</b> Hybrid (QuickSort + HeapSort + InsertionSort).
 * - <b>Stability:</b> Unstable
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N) (guaranteed by HeapSort fallback)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(log N) recursion stack space
 */
struct IntroSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     *
     * @return "introsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "introsort";
    }

    /**
     * @brief Sorts the span in-place using introsort.
     *
     * @tparam T Element type supporting @c operator<.
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> constexpr void sort(std::span<T> data) const {
        introsort(data);
    }

    /**
     * @brief Sorts the span in-place using introsort with a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator callable.
     * @param data Contiguous span of elements to sort.
     * @param comp Strict weak ordering comparator.
     */
    template <typename T, typename Compare>
    constexpr void sort(std::span<T> data, Compare comp) const {
        introsort(data, comp);
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "introsort", ::algoat::sorting::IntroSort)
