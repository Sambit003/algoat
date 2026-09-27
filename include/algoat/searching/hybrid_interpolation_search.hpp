/**
 * @file hybrid_interpolation_search.hpp
 * @brief Precision-Safe Hybrid Interpolation-Binary Search (IBS) engine.
 *
 * Implements a hybrid search algorithm combining exact 128-bit fixed-point interpolation
 * probing, adaptive logarithmic binary contraction budgeting, and branchless last-mile
 * cmov-predicated scans on small sub-ranges.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/searching/binary_search.hpp"

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

namespace algoat::searching {

namespace detail {

/**
 * @brief Executes a branchless cmov/adc-predicated linear scan over a leaf sub-range <= 16.
 *
 * Counts elements strictly less than target to determine the lower-bound index branchlessly,
 * eliminating pipeline stalls and sequential pointer-chasing data hazards on small windows.
 */
template <typename T>
[[nodiscard]] inline std::optional<std::size_t>
branchless_last_mile_scan(std::span<const T> data, std::size_t low, std::size_t high,
                          const T& target) noexcept {
    if (low > high || high >= data.size()) {
        return std::nullopt;
    }

    const std::size_t count = high - low + 1;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        offset += static_cast<std::size_t>(data[low + i] < target);
    }

    const std::size_t idx = low + offset;
    if (idx <= high && data[idx] == target) {
        return idx;
    }

    return std::nullopt;
}

/**
 * @brief Computes the interpolation iteration budget: B = log2(log2(N)) + 3.
 */
[[nodiscard]] constexpr std::size_t compute_contraction_budget(std::size_t len) noexcept {
    if (len <= 1) {
        return 3;
    }
    const std::size_t log_n = static_cast<std::size_t>(std::bit_width(len) - 1);
    const std::size_t log_log_n =
        (log_n > 0) ? static_cast<std::size_t>(std::bit_width(log_n) - 1) : 0;
    return log_log_n + 3;
}

} // namespace detail

/**
 * @brief Searches for target using Precision-Safe Hybrid Interpolation-Binary Search (IBS).
 *
 * Combines 128-bit exact integer arithmetic to avoid IEEE-754 floating-point truncation,
 * floating-point interpolation for float/double, an adaptive binary contraction budget to
 * prevent pathological O(N) clustering degradation, and a branchless linear scan on sub-ranges
 * <= 16.
 *
 * @tparam T Value type satisfying std::totally_ordered.
 *
 * @param data Sorted span of elements.
 * @param target Value to locate.
 * @return Index of a matching element if present, or std::nullopt.
 */
template <std::totally_ordered T>
[[nodiscard]] std::optional<std::size_t> hybrid_interpolation_search(std::span<const T> data,
                                                                     const T& target) noexcept {
    if (data.empty()) {
        return std::nullopt;
    }

    if constexpr (!std::is_arithmetic_v<T>) {
        // Fallback for non-arithmetic totally-ordered types (e.g., strings)
        // Delegates directly to BinarySearch to eliminate code duplication
        return BinarySearch{}.search(data, target);
    } else {
        std::size_t low = 0;
        std::size_t high = data.size() - 1;

        if (target < data[low] || target > data[high]) {
            return std::nullopt;
        }

        std::size_t budget = detail::compute_contraction_budget(data.size());
        std::size_t target_window = (high - low + 1) / 2;

        while (low <= high && target >= data[low] && target <= data[high]) {
            const std::size_t window_size = high - low + 1;

            if (window_size <= 16) {
                return detail::branchless_last_mile_scan(data, low, high, target);
            }

            if (data[high] == data[low]) {
                if (data[low] == target) {
                    return low;
                }
                return std::nullopt;
            }

            std::size_t pos = 0;
            if (budget > 0) {
                if constexpr (std::is_integral_v<T> && sizeof(T) <= 8) {
                    // Exact 128-bit integer fixed-point slope arithmetic (zero IEEE-754 floating
                    // point)
                    unsigned __int128 num_diff = 0;
                    unsigned __int128 den = 0;

                    if constexpr (std::is_signed_v<T>) {
                        const __int128_t dt =
                            static_cast<__int128_t>(target) - static_cast<__int128_t>(data[low]);
                        const __int128_t dr = static_cast<__int128_t>(data[high]) -
                                              static_cast<__int128_t>(data[low]);
                        num_diff = static_cast<unsigned __int128>(dt);
                        den = static_cast<unsigned __int128>(dr);
                    } else {
                        num_diff = static_cast<unsigned __int128>(target) -
                                   static_cast<unsigned __int128>(data[low]);
                        den = static_cast<unsigned __int128>(data[high]) -
                              static_cast<unsigned __int128>(data[low]);
                    }

                    const unsigned __int128 span_len = static_cast<unsigned __int128>(high - low);
                    const unsigned __int128 offset = (num_diff * span_len) / den;
                    pos = low + static_cast<std::size_t>(offset);
                } else if constexpr (std::is_floating_point_v<T>) {
                    // Floating-point slope interpolation for float/double with bounds clamping
                    const double num_diff =
                        static_cast<double>(target) - static_cast<double>(data[low]);
                    const double den =
                        static_cast<double>(data[high]) - static_cast<double>(data[low]);
                    double ratio = (den != 0.0) ? (num_diff / den) : 0.0;
                    if (ratio < 0.0) {
                        ratio = 0.0;
                    }
                    if (ratio > 1.0) {
                        ratio = 1.0;
                    }
                    pos = low + static_cast<std::size_t>(ratio * static_cast<double>(high - low));
                } else {
                    pos = low + (high - low) / 2;
                }
                --budget;
            } else {
                // Adaptive binary fallback pivot to guarantee O(log N) worst-case
                pos = low + (high - low) / 2;
            }

            if (data[pos] == target) {
                return pos;
            }

            if (data[pos] < target) {
                low = pos + 1;
            } else {
                if (pos == 0) {
                    break;
                }
                high = pos - 1;
            }

            if (low > high) {
                break;
            }

            const std::size_t new_window = high - low + 1;
            if (new_window <= target_window) {
                target_window = new_window / 2;
                budget = detail::compute_contraction_budget(new_window);
            }
        }

        return std::nullopt;
    }
}

/**
 * @brief Overload for non-const spans.
 */
template <std::totally_ordered T>
[[nodiscard]] inline std::optional<std::size_t>
hybrid_interpolation_search(std::span<T> data, const T& target) noexcept {
    return hybrid_interpolation_search(std::span<const T>{data.data(), data.size()}, target);
}

/**
 * @struct HybridInterpolationSearch
 * @brief Search algorithm implementing the SearchAlgorithm concept.
 */
struct HybridInterpolationSearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "hybridinterpolationsearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "hybridinterpolationsearch";
    }

    template <typename T, typename Target = T>
    std::optional<std::size_t> search(std::span<const T> data, const Target& target) const {
        return hybrid_interpolation_search(data, static_cast<T>(target));
    }

    template <typename T, typename Target = T>
    std::optional<std::size_t> search(std::span<T> data, const Target& target) const {
        return hybrid_interpolation_search(std::span<const T>{data.data(), data.size()},
                                           static_cast<T>(target));
    }

    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }
};

} // namespace algoat::searching

ALGOAT_REGISTER_ALGORITHM("searching", "hybridinterpolationsearch",
                          ::algoat::searching::HybridInterpolationSearch)
