/**
 * @file introsort.hpp
 * @brief Introspective Sort (IntroSort) hybrid algorithm implementation.
 */

#pragma once

#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <functional>
#include <span>
#include <string_view>

namespace algoat::sorting {

/**
 * @struct IntroSort
 * @brief Hybrid sorting algorithm combining QuickSort, HeapSort, and InsertionSort.
 *
 * Developed by David Musser (1997), IntroSort begins with QuickSort and monitors
 * recursion depth. If the recursion depth exceeds <tt>2 * floor(log2(N))</tt>, it
 * switches to HeapSort to prevent quadratic worst-case degradation. Small partitions
 * (<tt><= 32</tt> elements) are sorted using InsertionSort.
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
namespace detail {

template <typename T, typename Compare> T min_val(const T& a, const T& b, Compare comp) {
    return comp(b, a) ? b : a;
}

template <typename T, typename Compare> T max_val(const T& a, const T& b, Compare comp) {
    return comp(a, b) ? b : a;
}

template <typename T, typename Compare>
T get_pivot(const T& a, const T& b, const T& c, Compare comp) {
    return max_val(min_val(a, b, comp), min_val(max_val(a, b, comp), c, comp), comp);
}

template <typename T, typename Compare>
void introsort_impl(std::span<T> data, int depth_limit, Compare comp) {
    if (data.size() <= 32) {
        insertionsort(data, comp);
        return;
    }

    if (depth_limit == 0) {
        heapsort(data, comp);
        return;
    }

    // Median-of-three pivot selection
    auto pivot = get_pivot(data[0], data[data.size() / 2], data[data.size() - 1], comp);

    // Hoare partition scheme
    auto* left = data.data() - 1;
    auto* right = data.data() + data.size();

    while (true) {
        do {
            left++;
        } while (comp(*left, pivot));
        do {
            right--;
        } while (comp(pivot, *right));
        if (left >= right)
            break;
        std::swap(*left, *right);
    }

    std::size_t pivot_idx = right - data.data() + 1;

    introsort_impl(data.subspan(0, pivot_idx), depth_limit - 1, comp);
    introsort_impl(data.subspan(pivot_idx, data.size() - pivot_idx), depth_limit - 1, comp);
}

} // namespace detail

/**
 * @brief Sorts the span in-place using introsort with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> void introsort(std::span<T> data, Compare comp) {
    if (data.empty())
        return;
    int depth_limit = 2 * std::bit_width(data.size());
    detail::introsort_impl(data, depth_limit, comp);
}

/**
 * @brief Sorts the span in-place using introsort.
 * @tparam T Element type supporting <tt>operator<</tt>.
 *
 * @param data Contiguous span of elements to sort.
 */
template <typename T> void introsort(std::span<T> data) {
    introsort(data, std::less<T>{});
}

/**
 * @struct IntroSort
 * @brief Hybrid sorting algorithm combining QuickSort, HeapSort, and InsertionSort.
 *
 * Developed by David Musser (1997), IntroSort begins with QuickSort and monitors
 * recursion depth. If the recursion depth exceeds <tt>2 * floor(log2(N))</tt>, it
 * switches to HeapSort to prevent quadratic worst-case degradation. Small partitions
 * (<tt><= 32</tt> elements) are sorted using InsertionSort.
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
     * @return "introsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "introsort";
    }

    /**
     * @brief Sorts the span in-place using introsort.
     * @tparam T Element type.
     *
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        introsort(data);
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
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
