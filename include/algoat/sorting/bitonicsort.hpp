/**
 * @file bitonicsort.hpp
 * @brief Bitonic Sort sorting network implementation.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <functional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace algoat::sorting {

namespace detail {

template <typename T, typename Compare> void bitonic_merge(std::span<T> a, bool dir, Compare comp) {
    if (a.size() > 1) {
        std::size_t k = a.size() / 2;
        for (std::size_t i = 0; i < k; ++i) {
            bool greater = comp(a[i + k], a[i]);
            if (dir == greater) {
                std::swap(a[i], a[i + k]);
            }
        }
        bitonic_merge(a.subspan(0, k), dir, comp);
        bitonic_merge(a.subspan(k, k), dir, comp);
    }
}

template <typename T, typename Compare>
void bitonic_sort_impl(std::span<T> a, bool dir, Compare comp) {
    if (a.size() > 1) {
        std::size_t k = a.size() / 2;
        bitonic_sort_impl(a.subspan(0, k), true, comp);
        bitonic_sort_impl(a.subspan(k, k), false, comp);
        bitonic_merge(a, dir, comp);
    }
}

} // namespace detail

/**
 * @brief Sorts the span in-place using bitonic sorting network with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 *
 * @param data Span of elements to sort (must be power of 2 size).
 * @param comp Strict weak ordering comparator.
 * @throws std::invalid_argument If @c data.size() is not a power of 2.
 */
template <typename T, typename Compare> void bitonicsort(std::span<T> data, Compare comp) {
    if (data.empty())
        return;
    if (!std::has_single_bit(data.size())) {
        throw std::invalid_argument("Bitonic sort requires array size to be a power of 2");
    }
    detail::bitonic_sort_impl(data, true, comp);
}

/**
 * @brief Sorts the span in-place using bitonic sorting network.
 *
 * @tparam T Element type supporting @c operator<.
 * @param data Span of elements to sort (must be power of 2 size).
 * @throws std::invalid_argument If @c data.size() is not a power of 2.
 */
template <typename T> void bitonicsort(std::span<T> data) {
    bitonicsort(data, std::less<T>{});
}

/**
 * @struct BitonicSort
 * @brief Parallel sorting network algorithm designed for power-of-two dataset sizes.
 *
 * Recursively creates bitonic sequences (sequences that first monotonically increase,
 * then monotonically decrease) and merges them. Requires the input span size to be a
 * power of 2 (@c N = 2^k).
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Sorting Network.
 * - <b>Stability:</b> Unstable.
 * - <b>Constraint:</b> Input size must satisfy @c std::has_single_bit(N).
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log^2 N)
 * - Average Case: @c O(N log^2 N)
 * - Worst Case: @c O(N log^2 N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(log^2 N) recursion stack space
 */
struct BitonicSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     *
     * @return "bitonicsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "bitonicsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     *
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using bitonic sorting network.
     *
     * @tparam T Element type supporting @c operator<.
     * @param data Span of elements to sort (must be power of 2 size).
     * @throws std::invalid_argument If @c data.size() is not a power of 2.
     */
    template <typename T> void sort(std::span<T> data) const {
        bitonicsort(data);
    }

    /**
     * @brief Sorts the span in-place using bitonic sorting network with a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator callable.
     * @param data Span of elements to sort (must be power of 2 size).
     * @param comp Strict weak ordering comparator.
     * @throws std::invalid_argument If @c data.size() is not a power of 2.
     */
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        bitonicsort(data, comp);
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "bitonicsort", ::algoat::sorting::BitonicSort)
