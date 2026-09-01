/**
 * @file linear_search.hpp
 * @brief Linear Search algorithm implementation.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

namespace algoat::searching {

/**
 * @struct LinearSearch
 * @brief Sequential search scanning elements from left to right.
 *
 * Works on unsorted collections and returns the first occurrence of the target element.
 *
 * @par Characteristics:
 * - <b>Preconditions:</b> None (works on unsorted or sorted data).
 *
 * @par Time Complexity:
 * - Best Case: @c O(1) (target is at index 0)
 * - Average Case: @c O(N)
 * - Worst Case: @c O(N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct LinearSearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "linearsearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "linearsearch";
    }

    /**
     * @brief Linearly searches for target in the span.
     * @tparam T Element type supporting <tt>operator==</tt>.
     *
     * @param data Span of elements to search.
     *
     * @param target Value to locate.
     * @return Index of the first matching element, or @c std::nullopt.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }

    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        for (std::size_t i = 0; i < data.size(); ++i) {
            if (data[i] == target) {
                return i; // returns first match
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Indicates whether this search algorithm requires sorted input.
     * @return @c false
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return false;
    }
};

} // namespace algoat::searching
