/**
 * @file cyclesort.hpp
 * @brief In-place Cycle Sort minimizing memory write operations.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

/**
 * @brief Sorts the span in-place using cycle sort with a custom comparator.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator callable.
 * @param data Contiguous span of elements to sort.
 * @param comp Strict weak ordering comparator.
 */
template <typename T, typename Compare> void cyclesort(std::span<T> data, Compare comp) {
    if (data.size() <= 1)
        return;

    auto equiv = [&](const T& a, const T& b) { return !comp(a, b) && !comp(b, a); };

    for (std::size_t cycle_start = 0; cycle_start < data.size() - 1; ++cycle_start) {
        T item = std::move(data[cycle_start]);

        // Find where to put the item
        std::size_t pos = cycle_start;
        for (std::size_t i = cycle_start + 1; i < data.size(); ++i) {
            if (comp(data[i], item)) {
                pos++;
            }
        }

        // If the item is already in correct position, skip cycle
        if (pos == cycle_start) {
            data[cycle_start] = std::move(item);
            continue;
        }

        // Ignore duplicates
        while (equiv(item, data[pos])) {
            pos++;
        }

        if (pos != cycle_start) {
            std::swap(item, data[pos]);
        }

        while (pos != cycle_start) {
            pos = cycle_start;
            for (std::size_t i = cycle_start + 1; i < data.size(); ++i) {
                if (comp(data[i], item)) {
                    pos++;
                }
            }
            if (pos == cycle_start) {
                data[cycle_start] = std::move(item);
                break;
            }
            while (equiv(item, data[pos])) {
                pos++;
            }
            if (!equiv(item, data[pos])) {
                std::swap(item, data[pos]);
            }
        }
    }
}

/**
 * @brief Sorts the span in-place using cycle sort.
 *
 * @tparam T Element type supporting @c operator< and @c operator==.
 * @param data Contiguous span of elements to sort.
 */
template <typename T> void cyclesort(std::span<T> data) {
    cyclesort(data, std::less<T>{});
}

/**
 * @struct CycleSort
 * @brief Comparison sort that is theoretically optimal in total memory writes.
 *
 * Each element is either written zero times (if already in place) or exactly once
 * into its correct cyclic position. Ideal for Flash or EEPROM memory where write
 * operations degrade memory longevity.
 *
 * @par Characteristics:
 * - <b>Category:</b> Comparison-based, Selection/Cycle.
 * - <b>Stability:</b> Unstable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N^2)
 * - Average Case: @c O(N^2)
 * - Worst Case: @c O(N^2)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space (strictly in-place)
 */
struct CycleSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     *
     * @return "cyclesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "cyclesort";
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
     * @brief Sorts the span in-place using cycle sort.
     *
     * @tparam T Element type supporting @c operator< and @c operator==.
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        cyclesort(data);
    }

    /**
     * @brief Sorts the span in-place using cycle sort with a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator callable.
     * @param data Contiguous span of elements to sort.
     * @param comp Strict weak ordering comparator.
     */
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        cyclesort(data, comp);
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "cyclesort", ::algoat::sorting::CycleSort)
