/**
 * @file heapsort.hpp
 * @brief In-place comparison-based Heap Sort algorithm.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

/**
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
namespace detail {

/**
 * @brief Sifts down the subtree rooted at index @c i to maintain max-heap property.
 */
template <typename T, typename Compare>
void heapify(T* arr, std::size_t n, std::size_t i, Compare comp) {
    std::size_t largest = i;
    std::size_t left = 2 * i + 1;
    std::size_t right = 2 * i + 2;

    if (left < n && comp(arr[largest], arr[left])) {
        largest = left;
    }

    if (right < n && comp(arr[largest], arr[right])) {
        largest = right;
    }

    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest, comp);
    }
}

} // namespace detail

/**
 * @brief Sorts the span in-place using a binary max-heap with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> void heapsort(std::span<T> data, Compare comp) {
    const std::size_t n = data.size();
    if (n <= 1)
        return;

    // Build max-heap (bottom-up heap construction in O(n))
    for (std::size_t i = n / 2; i > 0; --i) {
        detail::heapify(data.data(), n, i - 1, comp);
    }

    // Extract maximum element one by one from root to the end
    for (std::size_t i = n - 1; i > 0; --i) {
        std::swap(data[0], data[i]);
        detail::heapify(data.data(), i, 0, comp);
    }
}

/**
 * @brief Sorts the span in-place using a binary max-heap.
 * @tparam T Element type supporting <tt>operator<</tt>.
 *
 * @param data Contiguous span of elements to sort.
 */
template <typename T> void heapsort(std::span<T> data) {
    heapsort(data, std::less<T>{});
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
     * @return "heapsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "heapsort";
    }

    /**
     * @brief Sorts the span in-place using a binary max-heap.
     * @tparam T Element type supporting <tt>operator<</tt>.
     *
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        heapsort(data);
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
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
