/**
 * @file linear_search.hpp
 * @brief Linear Search implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace algoat::searching {

/**
 * @struct LinearSearch
 * @brief Linear search algorithm.
 */
struct LinearSearch {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "linear_search";
    }

    /**
     * @brief Searches for value in span.
     * @tparam T Element type.
     * @param data Span to search in.
     * @param value Value to find.
     * @return Index of first occurrence, or npos if not found.
     */
    template <typename T> std::size_t search(std::span<T> data, const T& value) const {
        for (std::size_t i = 0; i < data.size(); ++i) {
            if (data[i] == value) {
                return i;
            }
        }
        return static_cast<std::size_t>(-1); // npos
    }
};

// ============ NEW: Free function ============
template <typename T> std::size_t linear_search(std::span<T> data, const T& value) {
    return LinearSearch{}.search(data, value);
}

} // namespace algoat::searching