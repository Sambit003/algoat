/**
 * @file insertionsort.hpp
 * @brief In-place comparison-based Insertion Sort algorithm.
 */

#pragma once

#include <string_view>
#include <span>
#include <cstddef>
#include <utility>

namespace algoat::sorting {

/**
 * @struct InsertionSort
 * @brief Standard stable Insertion Sort implementation with move semantics.
 * 
 * Efficient for small sequences (<tt>N < 32</tt>) and nearly sorted data. Used as
 * the default base-case sort in hybrid algorithms (IntroSort, TimSort, BlockSort).
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Insertion.
 * - <b>Stability:</b> Stable (preserves relative order of equal keys).
 *
 * @par Time Complexity:
 * - Best Case: @c O(N) (already sorted)
 * - Average Case: @c O(N^2)
 * - Worst Case: @c O(N^2) (reverse sorted)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space (in-place)
 */
struct InsertionSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "insertionsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "insertionsort";
    }

    /**
     * @brief Sorts the given span using insertion sort.
     * @tparam T Element type supporting <tt>operator></tt> and move construction/assignment.
 *
 * @param data Span of elements to sort in-place.
     */
    template<typename T>
    void sort(std::span<T> data) const {
        const std::size_t n = data.size();
        for (std::size_t i = 1; i < n; ++i) {
            T key = std::move(data[i]);
            std::size_t j = i;
            while (j > 0 && data[j - 1] > key) {
                data[j] = std::move(data[j - 1]);
                --j;
            }
            data[j] = std::move(key);
        }
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0 (effective from size 0 upwards).
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }
};

} // namespace algoat::sorting
