/**
 * @file adaptive_binary_search.hpp
 * @brief Adaptive Binary Search algorithm with dynamic monotonicity spot-checking and fallback.
 */

#pragma once

#include "algoat/searching/linear_search.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

namespace algoat::searching {

/**
 * @struct AdaptiveBinarySearch
 * @brief Sub-linear search on potentially sorted spans with dynamic invariant verification.
 *
 * Traverses bisection points in O(log N) time while spot-checking monotonicity constraints
 * and local invariants. If an invariant violation (such as an inversion) is detected,
 * it immediately aborts the logarithmic descent and safely pivots to LinearSearch in O(N) time.
 * Note: If no invariant violation is tripped during bisection, it will return std::nullopt.
 * Therefore, finding elements in fully unsorted datasets is not guaranteed.
 *
 * @par Characteristics:
 * - <b>Preconditions:</b> Data should ideally be sorted. Unsorted arrays are handled gracefully
 *   if an inversion is detected along the bisection path, but missing elements may return nullopt.
 *
 * @par Time Complexity:
 * - Best Case: @c O(1) (target is at midpoint or first inspected location)
 * - Sorted Average Case: @c O(log N) (pure bisection path)
 * - Unsorted / Worst Case: @c O(N) (graceful fallback)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct AdaptiveBinarySearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "adaptivebinarysearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "adaptivebinarysearch";
    }

    /**
     * @brief Searches for target using adaptive binary search with monotonicity verification.
     * @tparam T Element type supporting comparisons.
     *
     * @param data Span of elements to search.
     * @param target Value to locate.
     * @return Index of a matching element if present, or std::nullopt.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }

    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        if (data.empty()) {
            return std::nullopt;
        }

        const std::size_t n = data.size();
        if (n == 1) {
            return (data[0] == target) ? std::optional<std::size_t>{0} : std::nullopt;
        }

        // Fast endpoint check
        if (data.front() > data.back()) {
            return LinearSearch{}.search(data, target);
        }

        // Dynamic bounded bisection
        std::size_t low = 0;
        std::size_t high = n - 1;

        bool has_lower_bound = false;
        T lower_bound_val{};
        bool has_upper_bound = false;
        T upper_bound_val{};

        while (low <= high) {
            std::size_t mid = low + (high - low) / 2;
            const T& mid_val = data[mid];

            // Monotonicity invariant violation check
            if ((has_lower_bound && mid_val < lower_bound_val) ||
                (has_upper_bound && mid_val > upper_bound_val)) {
                return LinearSearch{}.search(data, target);
            }

            // Local adjacent pair spot-check
            if (mid + 1 < n && mid_val > data[mid + 1]) {
                return LinearSearch{}.search(data, target);
            }

            if (mid_val == target) {
                return mid;
            }

            if (mid_val < target) {
                lower_bound_val = mid_val;
                has_lower_bound = true;
                low = mid + 1;
            } else {
                upper_bound_val = mid_val;
                has_upper_bound = true;
                if (mid == 0) {
                    break;
                }
                high = mid - 1;
            }
        }

        // If target was not found in the bisection path:
        // Return std::nullopt to guarantee O(log N) latency for missing elements on sorted data.
        return std::nullopt;
    }

    /**
     * @brief Indicates whether this search algorithm requires sorted input.
     * @return @c false (adapts dynamically)
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return false;
    }
};

} // namespace algoat::searching
