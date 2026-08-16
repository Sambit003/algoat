/**
 * @file shellsort.hpp
 * @brief Shell Sort implementation using Ciura gap sequence.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <utility>
#include <iterator>

namespace algoat::sorting {

/**
 * @struct ShellSort
 * @brief Diminishing increment sorting algorithm utilizing the optimal Ciura gap sequence.
 * 
 * Generalizes insertion sort by comparing elements separated by decreasing gaps.
 * Uses Marcin Ciura's empirically optimal gap sequence (<tt>1, 4, 10, 23, 57, 132, 301, 701, ...</tt>),
 * providing superior practical runtime among comparison sorts.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Insertion with Gaps.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N^(4/3)) or @c O(N log^2 N) with Ciura gaps
 * - Worst Case: @c O(N^(3/2))
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct ShellSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "shellsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "shellsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using Shell Sort with Ciura gaps.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.size() <= 1) return;

        // Ciura gap sequence (empirically derived for optimal comparison counts)
        constexpr std::size_t gaps[] = {
            1, 4, 10, 23, 57, 132, 301, 701, 1577, 3548, 7983, 17961, 
            40412, 90927, 204585, 460316, 1035711, 2330350, 5243287, 
            11797395, 26544138, 59724310, 134379697, 302354318, 680297215
        };
        
        int gap_idx = std::size(gaps) - 1;
        while (gap_idx >= 0 && gaps[gap_idx] >= data.size()) {
            gap_idx--;
        }

        for (; gap_idx >= 0; gap_idx--) {
            std::size_t g = gaps[gap_idx];
            for (std::size_t i = g; i < data.size(); ++i) {
                T temp = std::move(data[i]);
                std::size_t j = i;
                while (j >= g && temp < data[j - g]) {
                    data[j] = std::move(data[j - g]);
                    j -= g;
                }
                data[j] = std::move(temp);
            }
        }
    }
};

} // namespace algoat::sorting
