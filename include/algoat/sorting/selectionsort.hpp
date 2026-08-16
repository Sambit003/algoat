/**
 * @file selectionsort.hpp
 * @brief Selection Sort comparison sorting algorithm.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>

namespace algoat::sorting {

/**
 * @struct SelectionSort
 * @brief Simple in-place comparison-based selection sort.
 * 
 * Divides the input into a sorted and unsorted region, repeatedly finding the
 * minimum element from the unsorted region and swapping it into the sorted boundary.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Selection.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N^2)
 * - Average Case: @c O(N^2)
 * - Worst Case: @c O(N^2)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct SelectionSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "selectionsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "selectionsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using Selection Sort.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        for (auto it = data.begin(); it != data.end(); ++it) {
            auto min_it = std::min_element(it, data.end());
            if (min_it != it) {
                std::iter_swap(it, min_it);
            }
        }
    }
};

} // namespace algoat::sorting
