/**
 * @file binary_search.hpp
 * @brief Binary Search algorithm implementation.
 */

#pragma once

#include <string_view>
#include <span>
#include <optional>
#include <cstddef>

namespace algoat::searching {

/**
 * @struct BinarySearch
 * @brief Logarithmic search on sorted spans using midpoint bisection.
 *
 * @par Characteristics:
 * - <b>Preconditions:</b> Range must be sorted in ascending order.
 *
 * @par Time Complexity:
 * - Best Case: @c O(1) (target is at midpoint)
 * - Average Case: @c O(log N)
 * - Worst Case: @c O(log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct BinarySearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "binarysearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "binarysearch";
    }

    /**
     * @brief Searches for target using binary search with overflow-safe midpoint calculation.
     * @tparam T Element type supporting <tt>operator==</tt> and <tt>operator<</tt>.
 *
 * @param data Sorted span of elements.
 *
 * @param target Value to locate.
     * @return Index of a matching element if present, or @c std::nullopt.
     */
    template<typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        if (data.empty()) return std::nullopt;

        std::size_t left = 0;
        std::size_t right = data.size() - 1;

        while (left <= right) {
            std::size_t mid = left + (right - left) / 2;
            if (data[mid] == target) {
                return mid;
            } else if (data[mid] < target) {
                left = mid + 1;
            } else {
                if (mid == 0) break;
                right = mid - 1;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Indicates whether this search algorithm requires sorted input.
     * @return @c true
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }
};

} // namespace algoat::searching
