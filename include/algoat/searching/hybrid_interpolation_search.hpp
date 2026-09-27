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
 * @brief Executes a branchless cmov-predicated lower-bound search over a leaf sub-range <= 16.
 *
 * Eliminates branch mispredictions and pipeline stalls on small candidate windows.
 */
template <typename T>
[[nodiscard]] inline std::optional<std::size_t>
branchless_last_mile_scan(std::span<const T> data, std::size_t low, std::size_t high,
                          const T& target) noexcept {
    if (low > high || high >= data.size()) {
        return std::nullopt;
    }

    const T* base = data.data() + low;
    std::size_t len = high - low + 1;

    while (len > 1) {
        std::size_t half = len / 2;
        base = (base[half] < target) ? (base + half) : base;
        len -= half;
    }

    if (*base == target) {
        return static_cast<std::size_t>(base - data.data());
    }
    if (base + 1 <= data.data() + high && *(base + 1) == target) {
        return static_cast<std::size_t>(base + 1 - data.data());
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
 * an adaptive binary contraction budget to prevent pathological O(N) clustering degradation,
 * and a branchless cmov scan on sub-ranges <= 16 elements.
 *
 * @tparam T Value type satisfying std::totally_ordered.
 * @tparam Extent Span extent (defaults to std::dynamic_extent).
 *
 * @param data Sorted span of elements.
 * @param target Value to locate.
 * @return Index of a matching element if present, or std::nullopt.
 */
template <std::totally_ordered T, typename Target = T, std::size_t Extent = std::dynamic_extent>
[[nodiscard]] std::optional<std::size_t>
hybrid_interpolation_search(std::span<const T, Extent> data, const Target& target_val) noexcept {
    if (data.empty()) {
        return std::nullopt;
    }

    const T target = static_cast<T>(target_val);

    // Branchless binary search fallback for non-integral or oversized types
    if constexpr (!std::is_integral_v<T> || sizeof(T) > 8) {
        const T* base = data.data();
        std::size_t range_length = data.size();

        while (range_length > 1) {
            std::size_t half = range_length / 2;
            base = (base[half] < target) ? (base + half) : base;
            range_length -= half;
        }

        const std::size_t index = static_cast<std::size_t>(base - data.data());
        if (*base == target) {
            return index;
        }
        if (index + 1 < data.size() && *(base + 1) == target) {
            return index + 1;
        }

        return std::nullopt;
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
                return detail::branchless_last_mile_scan(
                    std::span<const T>{data.data(), data.size()}, low, high, target);
            }

            if (data[high] == data[low]) {
                if (data[low] == target) {
                    return low;
                }
                return std::nullopt;
            }

            std::size_t pos = 0;
            if (budget > 0) {
                // Exact 128-bit integer fixed-point slope arithmetic (zero IEEE-754 floating point)
                unsigned __int128 num_diff = 0;
                unsigned __int128 den = 0;

                if constexpr (std::is_signed_v<T>) {
                    const __int128_t dt =
                        static_cast<__int128_t>(target) - static_cast<__int128_t>(data[low]);
                    const __int128_t dr =
                        static_cast<__int128_t>(data[high]) - static_cast<__int128_t>(data[low]);
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
template <std::totally_ordered T, typename Target = T, std::size_t Extent = std::dynamic_extent>
[[nodiscard]] inline std::optional<std::size_t>
hybrid_interpolation_search(std::span<T, Extent> data, const Target& target) noexcept {
    return hybrid_interpolation_search(std::span<const T, Extent>{data.data(), data.size()},
                                       target);
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

    template <typename T, typename Target = T, std::size_t Extent = std::dynamic_extent>
    std::optional<std::size_t> search(std::span<T, Extent> data, const Target& target) const {
        return search(std::span<const T, Extent>{data.data(), data.size()}, target);
    }

    template <typename T, typename Target = T, std::size_t Extent = std::dynamic_extent>
    std::optional<std::size_t> search(std::span<const T, Extent> data, const Target& target) const {
        return hybrid_interpolation_search(data, target);
    }

    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }
};

} // namespace algoat::searching

ALGOAT_REGISTER_ALGORITHM("searching", "hybridinterpolationsearch",
                          ::algoat::searching::HybridInterpolationSearch)
