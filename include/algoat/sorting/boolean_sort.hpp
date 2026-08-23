/**
 * @file boolean_sort.hpp
 * @brief High-performance branchless O(N) boolean array sorting.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <concepts>
#include <utility>

namespace algoat::sorting {

/**
 * @brief In-place Boolean-style partition for integral values.
 *
 * Moves all zero values before nonzero values while preserving every
 * original element. The relative order of elements is not guaranteed.
 *
 * For example:
 * @code
 * {0, 2, 0, 5} -> {0, 0, 2, 5}
 * @endcode
 *
 * @par Characteristics:
 * - <b>Category:</b> In-place two-way partition.
 * - <b>Time Complexity:</b> O(N).
 * - <b>Space Complexity:</b> O(1) auxiliary space.
 *
 * @tparam T An integral type.
 * @param data Contiguous span of integral values to partition in-place.
 * 
 */

template <std::integral T>
inline void sort_boolean(std::span<T> data) noexcept {
    std::size_t left = 0;
    std::size_t right = data.size();

    while(left<right){
        while(left<right && data[left]==0){
            left++;
        }
        while(left<right && data[right-1]!=0){
            right--;
        }

        if(left<right){
            std::swap(data[left],data[right-1]);
            left++;
            right--;
        }
    }
}

} // namespace algoat::sorting
