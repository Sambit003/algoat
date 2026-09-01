/**
 * @file binary_search.hpp
 * @brief Binary Search implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace algoat::searching {

/**
 * @struct BinarySearch
 * @brief Binary search algorithm on sorted ranges.
 */
struct BinarySearch {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "binary_search";
    }

    /**
     * @brief Searches for value in sorted span.
     * @tparam T Element type.
     * @param data Sorted span to search in.
     * @param value Value to find.
     * @return true if found, false otherwise.
     */
    template <typename T>
    bool search(std::span<T> data, const T& value) const {
        std::size_t left = 0;
        std::size_t right = data.size();

        while (left < right) {
            std::size_t mid = left + (right - left) / 2;
            if (data[mid] == value) {
                return true;
            } else if (data[mid] < value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return false;
    }
};

// ============ NEW: Free function ============
template <typename T>
bool binary_search(std::span<T> data, const T& value) {
    return BinarySearch{}.search(data, value);
}

} // namespace algoat::searching