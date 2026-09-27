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
 * @brief Executes a branchless cmov-predicated linear scan over a leaf sub-range <= 16.
 */
template <typename T>
[[nodiscard]] inline std::optional<std::size_t>
branchless_last_mile_scan(std::span<const T> window, const T& target) noexcept {
    if (window.empty()) {
        return std::nullopt;
    }

    std::size_t offset = 0;
    for (std::size_t i = 0; i < window.size(); ++i) {
        offset = (window[i] < target) ? offset + 1 : offset;
    }

    if (offset < window.size() && window[offset] == target) {
        return offset;
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

/**
 * @brief Converts generic compatible types into an unsigned 128-bit integer.
 */
template <typename T>
[[nodiscard]] constexpr unsigned __int128 to_u128(const T& val) noexcept
    requires(!std::is_floating_point_v<T>) && (std::is_pointer_v<T> || std::is_enum_v<T> ||
                                               requires { static_cast<unsigned __int128>(val); })
{
    if constexpr (std::is_pointer_v<T>) {
        return static_cast<unsigned __int128>(reinterpret_cast<uintptr_t>(val));
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<unsigned __int128>(static_cast<std::underlying_type_t<T>>(val));
    } else {
        return static_cast<unsigned __int128>(val);
    }
}

template <typename T>
concept interpolation_compatible = requires(const T& val) {
    { to_u128(val) } -> std::same_as<unsigned __int128>;
};

} // namespace detail

/**
 * @brief Searches for target using Precision-Safe Hybrid Interpolation-Binary Search (IBS).
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
    static_assert(
        detail::interpolation_compatible<T>,
        "hybrid_interpolation_search requires integer-like keys (uint64_t, pointers, timestamps).");

    if (data.empty()) {
        return std::nullopt;
    }

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
            auto res = detail::branchless_last_mile_scan(data.subspan(low, window_size), target);
            if (res) {
                return low + *res;
            }
            return std::nullopt;
        }

        if (data[high] == data[low]) {
            if (data[low] == target) {
                return low;
            }
            return std::nullopt;
        }

        std::size_t pos = 0;
        if (budget > 0) {
            // Exact 128-bit unsigned integer fixed-point slope arithmetic
            const unsigned __int128 diff_target =
                detail::to_u128(target) - detail::to_u128(data[low]);
            const unsigned __int128 diff_range =
                detail::to_u128(data[high]) - detail::to_u128(data[low]);

            const unsigned __int128 span_len = static_cast<unsigned __int128>(high - low);
            const unsigned __int128 offset = (diff_target * span_len) / diff_range;
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

/**
 * @struct HybridInterpolationSearch
 * @brief Search algorithm implementing the SearchAlgorithm concept for Registry.
 */
struct HybridInterpolationSearch {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "hybridinterpolationsearch";
    }

    template <typename T, typename Target = T>
        requires detail::interpolation_compatible<T>
    std::optional<std::size_t> search(std::span<const T> data, const Target& target) const {
        return hybrid_interpolation_search(data, static_cast<T>(target));
    }

    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }
};

} // namespace algoat::searching

ALGOAT_REGISTER_ALGORITHM("searching", "hybridinterpolationsearch",
                          ::algoat::searching::HybridInterpolationSearch)
