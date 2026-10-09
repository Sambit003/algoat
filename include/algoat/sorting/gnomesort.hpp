/**
 * @file gnomesort.hpp
 * @brief Gnome Sort implementation with last-position tracking optimization.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <span>
#include <string_view>

namespace algoat::sorting {

/**
 * @brief Sorts the span in-place using Gnome Sort with a custom comparator.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> void gnomesort(std::span<T> data, Compare comp) {
    std::size_t pos = 1;
    std::size_t last = 1;

    while (pos < data.size()) {
        if (pos == 0 || !comp(data[pos], data[pos - 1])) {
            pos = last;
            last++;
        } else {
            std::swap(data[pos], data[pos - 1]);
            pos--;
        }
    }
}

/**
 * @brief Sorts the span in-place using Gnome Sort.
 *
 * @tparam T Element type supporting @c operator<.
 * @param data Contiguous span of elements to sort.
 */
template <typename T> void gnomesort(std::span<T> data) {
    gnomesort(data, std::less<T>{});
}

/**
 * @struct GnomeSort
 * @brief Simple comparison sort similar to insertion sort using stepwise moves.
 *
 * Named after the garden gnome moving flower pots. Uses a @c last variable to track
 * the furthest advanced position, avoiding redundant linear traversals after backward swaps.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Exchange / Insertion.
 * - <b>Stability:</b> Stable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N)
 * - Average Case: @c O(N^2)
 * - Worst Case: @c O(N^2)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct GnomeSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     *
     * @return "gnomesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "gnomesort";
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
     * @brief Sorts the span in-place using Gnome Sort with position memorization.
     *
     * @tparam T Element type supporting @c operator<.
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        gnomesort(data);
    }

    /**
     * @brief Sorts the span in-place using Gnome Sort with a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator callable.
     * @param data Contiguous span of elements to sort.
     * @param comp Strict weak ordering comparator.
     */
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        gnomesort(data, comp);
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "gnomesort", ::algoat::sorting::GnomeSort)
