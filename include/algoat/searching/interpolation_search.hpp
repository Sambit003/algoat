/**
 * @file interpolation_search.hpp
 * @brief Interpolation Search algorithm implementation with binary search fallback.
 */

#pragma once

#include "algoat/utils.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

namespace algoat::searching {

/**
 * @struct InterpolationSearch
 * @brief Search algorithm estimating probe position based on key values in uniformly distributed
 * data.
 *
 * Uses the linear interpolation formula for arithmetic types:
 *
 * @par Characteristics:
 * - <b>Preconditions:</b> Range must be sorted in ascending order and ideally uniformly
 * distributed.
 *
 * @par Time Complexity:
 * - Best Case: @c O(1)
 * - Average Case: @c O(log(log(N))) (for uniformly distributed arithmetic data)
 * - Worst Case: @c O(N) (for exponentially distributed data)
 *
 * @par Space Complexity:
 * - Auxiliary Space: * - Auxiliary Space: @c O(1)
 */
struct InterpolationSearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "interpolationsearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "interpolationsearch";
    }

    /**
     * @brief Searches for target using arithmetic interpolation (or binary search fallback).
     * @tparam T Element type.
     *
     * @param[in] data Sorted span of elements.
     *
     * @param[in] target Value to locate.
     * @return Index of a matching element if present, or @c std::nullopt.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }

    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        if (data.empty())
            return std::nullopt;

        if constexpr (std::is_arithmetic_v<T>) {
            std::size_t low = 0;
            std::size_t high = data.size() - 1;

            while (low <= high && target >= data[low] && target <= data[high]) {
                if (low == high) {
                    if (data[low] == target)
                        return low;
                    return std::nullopt;
                }

                if (data[high] == data[low]) {
                    if (data[low] == target)
                        return low;
                    return std::nullopt;
                }

                double pos_double =
                    static_cast<double>(low) +
                    ((static_cast<double>(high) - static_cast<double>(low)) /
                     (static_cast<double>(data[high]) - static_cast<double>(data[low]))) *
                        (static_cast<double>(target) - static_cast<double>(data[low]));

                pos_double =
                    algoat::clamp(pos_double, static_cast<double>(low), static_cast<double>(high));
                std::size_t pos = static_cast<std::size_t>(pos_double);

                if (data[pos] == target) {
                    return pos;
                }

                if (data[pos] < target) {
                    low = pos + 1;
                } else {
                    if (pos == 0)
                        break;
                    high = pos - 1;
                }
            }
        } else {
            // Fallback to binary search for non-arithmetic types
            std::size_t left = 0;
            std::size_t right = data.size() - 1;
            while (left <= right) {
                std::size_t mid = left + (right - left) / 2;
                if (data[mid] == target)
                    return mid;
                if (data[mid] < target)
                    left = mid + 1;
                else {
                    if (mid == 0)
                        break;
                    right = mid - 1;
                }
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
