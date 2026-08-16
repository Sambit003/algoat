/**
 * @file combsort.hpp
 * @brief Comb Sort comparison sorting algorithm.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>

namespace algoat::sorting {

/**
 * @struct CombSort
 * @brief Improvement over Bubble Sort using a geometric shrink factor of 1.3.
 * 
 * Eliminates "turtles" (small values near the end of the array) by starting with a large
 * comparison gap and shrinking by ~1.3 each pass until .
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Exchange.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N^2 / 2^p) where @c p is the number of increments
 * - Worst Case: @c O(N^2)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct CombSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "combsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "combsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using Comb Sort.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        std::size_t gap = data.size();
        bool swapped = true;
        
        while (gap > 1 || swapped) {
            gap = (gap * 10) / 13; // Shrink factor 1.3
            if (gap < 1) gap = 1;
            
            swapped = false;
            for (std::size_t i = 0; i + gap < data.size(); ++i) {
                if (data[i + gap] < data[i]) {
                    std::swap(data[i], data[i + gap]);
                    swapped = true;
                }
            }
        }
    }
};

} // namespace algoat::sorting
