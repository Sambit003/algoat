/**
 * @file combsort.hpp
 * @brief Comb Sort comparison sorting algorithm.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <span>
#include <string_view>

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
/**
 * @brief Sorts the span in-place using Comb Sort with a custom comparator.
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 *
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> void combsort(std::span<T> data, Compare comp) {
    std::size_t gap = data.size();
    bool swapped = true;

    while (gap > 1 || swapped) {
        gap = (gap * 10) / 13; // Shrink factor 1.3
        if (gap < 1)
            gap = 1;

        swapped = false;
        for (std::size_t i = 0; i + gap < data.size(); ++i) {
            if (comp(data[i + gap], data[i])) {
                std::swap(data[i], data[i + gap]);
                swapped = true;
            }
        }
    }
}

/**
 * @brief Sorts the span in-place using Comb Sort.
 * @tparam T Element type supporting <tt>operator<</tt>.
 *
 * @param data Contiguous span of elements to sort.
 */
template <typename T> void combsort(std::span<T> data) {
    combsort(data, std::less<T>{});
}

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
     * @tparam T Element type.
     *
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        combsort(data);
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        combsort(data, comp);
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "combsort", ::algoat::sorting::CombSort)
