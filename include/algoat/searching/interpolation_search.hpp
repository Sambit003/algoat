/**
 * @file interpolation_search.hpp
 * @brief Interpolation Search implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace algoat::searching {

/**
 * @struct InterpolationSearch
 * @brief Interpolation search algorithm for uniformly distributed sorted data.
 */
struct InterpolationSearch {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "interpolation_search";
    }

    /**
     * @brief Searches for value in uniformly distributed sorted span.
     * @tparam T Element type.
     * @param data Sorted span to search in.
     * @param value Value to find.
     * @return Index of value, or npos if not found.
     */
    template <typename T>
    std::size_t search(std::span<T> data, const T& value) const {
        std::size_t low = 0;
        std::size_t high = data.size() - 1;

        while (low <= high && value >= data[low] && value <= data[high]) {
            if (low == high) {
                return (data[low] == value) ? low : static_cast<std::size_t>(-1);
            }

            // Interpolation formula
            std::size_t pos = low + static_cast<std::size_t>(
                (static_cast<double>(high - low) / (data[high] - data[low])) *
                (value - data[low])
            );

            if (data[pos] == value) {
                return pos;
            }

            if (data[pos] < value) {
                low = pos + 1;
            } else {
                high = pos - 1;
            }
        }
        return static_cast<std::size_t>(-1);
    }
};

// ============ NEW: Free function ============
template <typename T>
std::size_t interpolation_search(std::span<T> data, const T& value) {
    return InterpolationSearch{}.search(data, value);
}

} // namespace algoat::searching