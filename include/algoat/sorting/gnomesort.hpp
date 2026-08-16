/**
 * @file gnomesort.hpp
 * @brief Gnome Sort implementation with last-position tracking optimization.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>

namespace algoat::sorting {

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
     * @return "gnomesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "gnomesort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using Gnome Sort with position memorization.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        std::size_t pos = 1;
        std::size_t last = 1;
        
        while (pos < data.size()) {
            if (pos == 0 || data[pos - 1] <= data[pos]) {
                pos = last;
                last++;
            } else {
                std::swap(data[pos], data[pos - 1]);
                pos--;
            }
        }
    }
};

} // namespace algoat::sorting
