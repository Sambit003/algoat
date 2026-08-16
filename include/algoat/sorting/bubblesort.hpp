/**
 * @file bubblesort.hpp
 * @brief Bubble Sort implementation with early-termination optimization.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <utility>

namespace algoat::sorting {

/**
 * @struct BubbleSort
 * @brief Simple comparison-based exchange sorting algorithm.
 * 
 * Repeatedly steps through the list, compares adjacent elements, and swaps them
 * if they are in the wrong order. Includes the @c swapped boolean flag for @c O(N)
 * early termination if the array becomes sorted before all passes finish.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Exchange.
 * - <b>Stability:</b> Stable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N) (already sorted, terminates after 1 pass)
 * - Average Case: @c O(N^2)
 * - Worst Case: @c O(N^2)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct BubbleSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "bubblesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "bubblesort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using Bubble Sort.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.empty()) return;
        bool swapped = true;
        for (std::size_t i = 0; i < data.size() - 1 && swapped; ++i) {
            swapped = false;
            for (std::size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j + 1] < data[j]) {
                    std::swap(data[j], data[j + 1]);
                    swapped = true;
                }
            }
        }
    }
};

} // namespace algoat::sorting
