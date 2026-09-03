/**
 * @file boolean_sort.hpp
 * @brief High-performance branchless O(N) boolean array sorting.
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace algoat::sorting {

namespace detail {
// Hack 1: P3701R0 (May 2025 WG21 Proposal) "strict_integer"
// Standard std::integral conflates integers, characters, and booleans.
// We strictly isolate mathematical integers to prevent accidental character sorting.
template <typename T>
concept strict_integer =
    std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool> &&
    !std::same_as<std::remove_cv_t<T>, char> && !std::same_as<std::remove_cv_t<T>, wchar_t> &&
    !std::same_as<std::remove_cv_t<T>, char8_t> && !std::same_as<std::remove_cv_t<T>, char16_t> &&
    !std::same_as<std::remove_cv_t<T>, char32_t>;
} // namespace detail

/**
 * @brief Ultra-fast O(N) sorting for boolean 0/1 arrays.
 *
 * OPTIMIZATION: Concrete non-template function! (O(1) resolution)
 *
 * Performs a single pass counting zeros, followed by hardware-accelerated
 * @c std::memset calls.
 *
 * @param data Contiguous span of @c bool values to sort in-place.
 */
inline void sort_boolean(std::span<bool> data) noexcept {
    if (data.empty())
        return;

    size_t count_false = 0;
    for (bool val : data) {
        count_false += !val;
    }
    size_t count_true = data.size() - count_false;
    if (count_false > 0) {
        std::memset(data.data(), 0, count_false);
    }
    if (count_true > 0) {
        std::memset(data.data() + count_false, 1, count_true);
    }
}

/**
 * @brief Ultra-fast O(N) partitioning for strictly mathematical integral arrays.
 *
 * This overload handles non-boolean integral types via a branchless
 * 2-way in-place partition (Alexandrescu-Peters Lomuto scheme),
 * strictly preserving exact multiset values.
 *
 * @param data Contiguous span of non-boolean integral values to partition in-place.
 */
template <detail::strict_integer T> inline void sort_boolean(std::span<T> data) noexcept {
    if (data.empty())
        return;

    T* first = data.data();
    T* last = first + data.size();
    T* out = first;

    for (T* it = first; it != last; ++it) {
        const T val = *it;
        const bool is_zero = (val == 0);
        *it = *out;
        *out = val;
        out += is_zero;
    }
}

/**
 * @brief Hack 2: Constrained Poison Sink ("Anti-SFINAE")
 *
 * Hijacks unintended types (floats, characters) that fail the strict integer concept.
 * Instead of falling back to a generic SFINAE "no matching function" error,
 * this catches them and emits an explicitly deleted function error, preventing
 * silent decays or confusing generic failures.
 */
template <typename T>
    requires(!detail::strict_integer<T> && !std::same_as<std::remove_cv_t<T>, bool>)
void sort_boolean(std::span<T> data) = delete;

} // namespace algoat::sorting
