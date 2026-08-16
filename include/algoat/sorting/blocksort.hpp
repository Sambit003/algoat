/**
 * @file blocksort.hpp
 * @brief Block Sort (sqrt-decomposition merge sort) implementation.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <algorithm>
#include <cmath>
#include "algoat/sorting/insertionsort.hpp"

namespace algoat::sorting {

/**
 * @struct BlockSort
 * @brief Block Merge Sort utilizing  block decomposition.
 * 
 * Partitions the data into blocks of size ~sqrt(N), sorts each block
 * using @c std::sort, and merges adjacent blocks using in-place merge passes.
 *
 * @par Characteristics:
 * - <b>Category:</b> Hybrid, Block-based Merge Sort.
 * - <b>Stability:</b> Dependent on merge stability.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N log N)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) to @c O(N) during in-place merges
 */
struct BlockSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "blocksort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "blocksort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts the span in-place using block merge sort.
     * @tparam T Type satisfying @c std::totally_ordered.
 *
 * @param data Contiguous span of elements to sort.
     */
    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.empty()) return;
        
        std::size_t n = data.size();
        std::size_t block_size = static_cast<std::size_t>(std::sqrt(n));
        
        // Small block size fallback to InsertionSort
        if (block_size < 16) {
            InsertionSort{}.sort(data);
            return;
        }

        // Sort individual sqrt(N) blocks
        for (std::size_t i = 0; i < n; i += block_size) {
            std::size_t end = std::min(i + block_size, n);
            std::sort(data.begin() + i, data.begin() + end);
        }

        // Iteratively merge sorted blocks
        for (std::size_t size = block_size; size < n; size = 2 * size) {
            for (std::size_t left = 0; left < n; left += 2 * size) {
                std::size_t mid = std::min(left + size, n);
                std::size_t right = std::min(left + 2 * size, n);
                if (mid < right) {
                    std::inplace_merge(data.begin() + left, data.begin() + mid, data.begin() + right);
                }
            }
        }
    }
};

} // namespace algoat::sorting
