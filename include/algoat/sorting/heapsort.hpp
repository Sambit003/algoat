/**
 * @file heapsort.hpp
 * @brief In-place comparison-based Heap Sort algorithm.
 */

#pragma once

#include "algoat/core/traits.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail {

/**
 * @brief Sifts down the subtree rooted at index @c i to maintain max-heap property.
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 * @param first Beginning of the range.
 * @param n Total number of elements in the heap.
 * @param i Root index of the subtree to sift down.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void heapify(I first, std::size_t n, std::size_t i, Comp comp, Proj proj) {
    std::size_t largest = i;
    std::size_t left = 2 * i + 1;
    std::size_t right = 2 * i + 2;

    if (left < n && std::invoke(comp, std::invoke(proj, *(first + largest)),
                                std::invoke(proj, *(first + left)))) {
        largest = left;
    }

    if (right < n && std::invoke(comp, std::invoke(proj, *(first + largest)),
                                 std::invoke(proj, *(first + right)))) {
        largest = right;
    }

    if (largest != i) {
        std::ranges::iter_swap(first + i, first + largest);
        heapify(first, n, largest, comp, proj);
    }
}

/**
 * @brief Sifts down the subtree rooted at index @c i for raw pointer spans.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param arr Pointer to the start of the array.
 * @param n Total number of elements in the heap.
 * @param i Root index of the subtree to sift down.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare>
constexpr void heapify(T* arr, std::size_t n, std::size_t i, Compare comp) {
    heapify(arr, n, i, comp, std::identity{});
}

} // namespace detail

/**
 * @brief Sorts a range delimited by iterators [first, last) in-place using a binary max-heap with
 * custom comparator and projection.
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
constexpr void heapsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    const std::size_t n = static_cast<std::size_t>(last_it - first);
    if (n <= 1) {
        return;
    }
    for (std::size_t i = n / 2; i > 0; --i) {
        detail::heapify(first, n, i - 1, comp, proj);
    }
    for (std::size_t i = n - 1; i > 0; --i) {
        std::ranges::iter_swap(first, first + i);
        detail::heapify(first, i, 0, comp, proj);
    }
}

/**
 * @brief Sorts a C++20 random-access range in-place using a binary max-heap with custom comparator
 * and projection.
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
constexpr std::ranges::borrowed_iterator_t<R> heapsort(R&& data, Comp comp = {}, Proj proj = {}) {
    heapsort(std::ranges::begin(data), std::ranges::end(data), comp, proj);
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

/**
 * @brief Sorts the span in-place using a binary max-heap with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> constexpr void heapsort(std::span<T> data, Compare comp) {
    heapsort(data.begin(), data.end(), comp);
}

/**
 * @brief Sorts the span in-place using a binary max-heap.
 *
 * @tparam T Element type supporting @c operator<.
 * @param data Contiguous span of elements to sort.
 */
template <typename T> constexpr void heapsort(std::span<T> data) {
    heapsort(data.begin(), data.end(), std::less<T>{});
}

/**
 * @struct HeapSort
 * @brief In-place sorting algorithm utilizing a binary max-heap.
 *
 * Guaranteed @c O(N log N) worst-case performance with zero dynamic allocation.
 * Serves as the primary guaranteed fallback in @c algoat::core::Dispatcher and the
 * worst-case recursion safeguard in @c algoat::sorting::IntroSort.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Selection/Heap.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space (strictly in-place)
 */
struct HeapSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     *
     * @return "heapsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "heapsort";
    }

    /**
     * @brief Sorts the span in-place using a binary max-heap.
     *
     * @tparam T Element type supporting @c operator<.
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> constexpr void sort(std::span<T> data) const {
        heapsort(data);
    }

    /**
     * @brief Sorts the span in-place using a binary max-heap with a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator callable.
     * @param data Contiguous span of elements to sort.
     * @param comp Strict weak ordering comparator.
     */
    template <typename T, typename Compare>
    constexpr void sort(std::span<T> data, Compare comp) const {
        heapsort(data, comp);
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0 (ideal fallback algorithm across all sizes).
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0; // Ideal fallback
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "heapsort", ::algoat::sorting::HeapSort)
