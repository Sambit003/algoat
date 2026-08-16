/**
 * @file timsort.hpp
 * @brief Simplified TimSort hybrid sorting algorithm.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>
#include "algoat/sorting/insertionsort.hpp"

namespace algoat::sorting {

/**
 * @struct TimSort
 * @brief Adaptive stable hybrid sorting algorithm derived from Merge Sort and Insertion Sort.
 * 
 * Divides data into 32-element runs, sorts each run using @c InsertionSort, and then
 * iteratively merges runs using @c std::inplace_merge. Highly efficient for nearly sorted
 * data where run sortedness can be exploited.
 *
 * @par Characteristics:
 * - <b>Category:</b> Hybrid (Insertion Sort + Merge Sort), Adaptive.
 * - <b>Stability:</b> Stable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N) (already sorted)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N) during in-place merges
 */
struct TimSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "timsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "timsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /// Default run partition size for initial insertion sort phase.
    static constexpr std::size_t RUN_SIZE = 32;

    /**
     * @brief Sorts the span in-place using TimSort.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.empty()) return;
        
        std::size_t n = data.size();

        // 1. Sort individual runs of size RUN_SIZE with InsertionSort
        for (std::size_t i = 0; i < n; i += RUN_SIZE) {
            std::size_t end = std::min(i + RUN_SIZE, n);
            InsertionSort{}.sort(data.subspan(i, end - i));
        }

        // 2. Bottom-up merge sorted runs
        for (std::size_t size = RUN_SIZE; size < n; size = 2 * size) {
            for (std::size_t left = 0; left < n; left += 2 * size) {
                std::size_t mid = std::min(left + size, n);
                std::size_t right = std::min(left + 2 * size, n);

                if (mid < right) {
                    std::inplace_merge(
                        data.begin() + left,
                        data.begin() + mid,
                        data.begin() + right
                    );
                }
            }
        }
    }
};

} // namespace algoat::sorting
