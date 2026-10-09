/**
 * @file mergesort.hpp
 * @brief Stable Merge Sort implementation using auxiliary buffer.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory_resource>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::sorting {

namespace detail {

template <typename T, typename Compare>
void merge_impl(T* arr, T* temp, std::size_t left, std::size_t mid, std::size_t right,
                Compare comp) {
    std::size_t i = left;
    std::size_t j = mid + 1;
    std::size_t k = left;

    while (i <= mid && j <= right) {
        if (!comp(arr[j], arr[i])) { // <= ensures stability
            temp[k++] = std::move(arr[i++]);
        } else {
            temp[k++] = std::move(arr[j++]);
        }
    }

    while (i <= mid) {
        temp[k++] = std::move(arr[i++]);
    }

    while (j <= right) {
        temp[k++] = std::move(arr[j++]);
    }

    for (std::size_t p = left; p <= right; ++p) {
        arr[p] = std::move(temp[p]);
    }
}

template <typename T, typename Compare>
void mergesort_impl(T* arr, T* temp, std::size_t left, std::size_t right, Compare comp) {
    if (left < right) {
        std::size_t mid = left + (right - left) / 2;
        mergesort_impl(arr, temp, left, mid, comp);
        mergesort_impl(arr, temp, mid + 1, right, comp);
        merge_impl(arr, temp, left, mid, right, comp);
    }
}

} // namespace detail

/**
 * @brief Sorts the span in-place using top-down merge sort with a custom comparator and optional
 * memory resource.
 * @tparam T Element type supporting move operations.
 * @tparam Compare Strict weak ordering callable.
 *
 * @param data Span of elements to sort.
 * @param comp Custom comparator callable.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 */
template <typename T, typename Compare>
    requires(!std::is_convertible_v<Compare, std::pmr::memory_resource*>)
void mergesort(std::span<T> data, Compare comp,
               std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    if (data.size() <= 1)
        return;

    // Add padding to avoid a second allocation due to alignment offsets.
    std::size_t buffer_size = data.size() * sizeof(T) + alignof(T) * 4 + 64;
    std::pmr::monotonic_buffer_resource mbr(buffer_size, mr);

    if constexpr (std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_copy_assignable_v<T> && std::is_trivially_destructible_v<T>) {
        // Bypass O(N) default initialization overhead for trivial types
        std::pmr::polymorphic_allocator<T> alloc(&mbr);
        T* buffer = alloc.allocate(data.size());
        detail::mergesort_impl(data.data(), buffer, 0, data.size() - 1, comp);
    } else {
        std::pmr::vector<T> buffer(data.size(), &mbr);
        detail::mergesort_impl(data.data(), buffer.data(), 0, data.size() - 1, comp);
    }
}

/**
 * @brief Sorts the span in-place using top-down merge sort with an optional memory resource.
 * @tparam T Element type supporting @c operator<= and move operations.
 *
 * @param data Span of elements to sort.
 * @param mr Pointer to a polymorphic memory resource for the scratchpad buffer.
 */
template <typename T>
void mergesort(std::span<T> data,
               std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    mergesort(data, std::less<>{}, mr);
}

/**
 * @struct MergeSort
 * @brief Top-down, divide-and-conquer stable sorting algorithm.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Divide & Conquer.
 * - <b>Stability:</b> Stable (preserves order of equivalent keys via @c <= merge comparison).
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N) auxiliary buffer space
 */
struct MergeSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "mergesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "mergesort";
    }

    /**
     * @brief Sorts the span in-place using top-down merge sort.
     * @tparam T Element type supporting @c operator<= and move operations.
     *
     * @param data Span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        mergesort(data);
    }

    /**
     * @brief Sorts the span in-place using top-down merge sort with a custom comparator.
     * @tparam T Element type supporting move operations.
     * @tparam Compare Strict weak ordering callable.
     *
     * @param data Span of elements to sort.
     * @param comp Custom comparator callable.
     */
    template <typename T, typename Compare>
        requires(!std::is_convertible_v<Compare, std::pmr::memory_resource*>)
    void sort(std::span<T> data, Compare comp) const {
        mergesort(data, std::move(comp));
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 32
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "mergesort", ::algoat::sorting::MergeSort)
