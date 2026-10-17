#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::strings {

namespace detail {

template <typename T> inline constexpr bool is_span_v = false;

template <typename T, std::size_t Extent>
inline constexpr bool is_span_v<std::span<T, Extent>> = true;

template <typename T> [[nodiscard]] constexpr std::string_view to_string_view(const T& s) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::string_view>) {
        return s;
    } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::string>) {
        return std::string_view(s.data(), s.size());
    } else if constexpr (std::is_convertible_v<const T&, std::string_view>) {
        return std::string_view(s);
    } else {
        return std::string_view(s.data(), s.size());
    }
}

template <typename T> [[nodiscard]] constexpr int char_at(const T& s, std::size_t depth) noexcept {
    auto sv = to_string_view(s);
    if (depth < sv.size()) {
        return static_cast<unsigned char>(sv[depth]);
    }
    return -1;
}

template <typename StringType>
[[nodiscard]] constexpr bool string_less_from_depth(const StringType& a, const StringType& b,
                                                    std::size_t depth) noexcept {
    auto sva = to_string_view(a);
    auto svb = to_string_view(b);
    const std::size_t sa = sva.size();
    const std::size_t sb = svb.size();
    if (depth < sa && depth < sb) {
        std::string_view rem_a = sva.substr(depth);
        std::string_view rem_b = svb.substr(depth);
        return rem_a < rem_b;
    }
    return sa < sb;
}

template <typename StringType>
void string_insertion_sort(std::span<StringType> strings, std::size_t depth) {
    const std::size_t n = strings.size();
    for (std::size_t i = 1; i < n; ++i) {
        StringType key = std::move(strings[i]);
        std::size_t j = i;
        while (j > 0 && string_less_from_depth(key, strings[j - 1], depth)) {
            strings[j] = std::move(strings[j - 1]);
            --j;
        }
        strings[j] = std::move(key);
    }
}

template <typename StringType>
[[nodiscard]] int median_of_three_char(const std::span<StringType>& strings, std::size_t a,
                                       std::size_t b, std::size_t c, std::size_t depth) noexcept {
    int va = char_at(strings[a], depth);
    int vb = char_at(strings[b], depth);
    int vc = char_at(strings[c], depth);

    if (va > vb) {
        std::swap(va, vb);
    }
    if (vb > vc) {
        std::swap(vb, vc);
    }
    if (va > vb) {
        std::swap(va, vb);
    }
    return vb;
}

template <typename StringType>
[[nodiscard]] int select_pivot_char(const std::span<StringType>& strings,
                                    std::size_t depth) noexcept {
    const std::size_t n = strings.size();
    if (n <= 8) {
        return char_at(strings[n / 2], depth);
    }
    const std::size_t mid = n / 2;
    if (n <= 40) {
        return median_of_three_char(strings, 0, mid, n - 1, depth);
    }
    const std::size_t step = n / 8;
    int m1 = median_of_three_char(strings, 0, step, step * 2, depth);
    int m2 = median_of_three_char(strings, mid - step, mid, mid + step, depth);
    int m3 = median_of_three_char(strings, n - 1 - step * 2, n - 1 - step, n - 1, depth);

    if (m1 > m2) {
        std::swap(m1, m2);
    }
    if (m2 > m3) {
        std::swap(m2, m3);
    }
    if (m1 > m2) {
        std::swap(m1, m2);
    }
    return m2;
}

template <typename StringType>
void multikey_quicksort_impl(std::span<StringType> strings, std::size_t depth) {
    while (strings.size() > 16) {
        const std::size_t n = strings.size();
        const int pivot = select_pivot_char(strings, depth);

        std::ptrdiff_t p = 0;
        std::ptrdiff_t q = static_cast<std::ptrdiff_t>(n) - 1;
        std::ptrdiff_t i = 0;
        std::ptrdiff_t j = static_cast<std::ptrdiff_t>(n) - 1;

        while (true) {
            while (i <= j) {
                int c = char_at(strings[static_cast<std::size_t>(i)], depth);
                if (c > pivot) {
                    break;
                }
                if (c == pivot) {
                    std::swap(strings[static_cast<std::size_t>(p)],
                              strings[static_cast<std::size_t>(i)]);
                    ++p;
                }
                ++i;
            }
            while (i <= j) {
                int c = char_at(strings[static_cast<std::size_t>(j)], depth);
                if (c < pivot) {
                    break;
                }
                if (c == pivot) {
                    std::swap(strings[static_cast<std::size_t>(j)],
                              strings[static_cast<std::size_t>(q)]);
                    --q;
                }
                --j;
            }
            if (i > j) {
                break;
            }
            std::swap(strings[static_cast<std::size_t>(i)], strings[static_cast<std::size_t>(j)]);
            ++i;
            --j;
        }

        std::ptrdiff_t eq_left_count = std::min(p, i - p);
        for (std::ptrdiff_t k = 0; k < eq_left_count; ++k) {
            std::swap(strings[static_cast<std::size_t>(k)],
                      strings[static_cast<std::size_t>(i - eq_left_count + k)]);
        }

        std::ptrdiff_t eq_right_count = std::min(static_cast<std::ptrdiff_t>(n) - 1 - q, q - j);
        for (std::ptrdiff_t k = 0; k < eq_right_count; ++k) {
            std::swap(strings[static_cast<std::size_t>(i + k)],
                      strings[static_cast<std::size_t>(static_cast<std::ptrdiff_t>(n) -
                                                       eq_right_count + k)]);
        }

        std::size_t left_size = static_cast<std::size_t>(i - p);
        std::size_t mid_size =
            static_cast<std::size_t>(p + (static_cast<std::ptrdiff_t>(n) - 1 - q));
        std::size_t right_size = n - left_size - mid_size;

        auto left_slice = strings.subspan(0, left_size);
        auto mid_slice = strings.subspan(left_size, mid_size);
        auto right_slice = strings.subspan(left_size + mid_size, right_size);

        const std::size_t effective_mid_size = (pivot != -1) ? mid_size : 0;

        if (left_slice.empty() && right_slice.empty()) {
            if (pivot == -1) {
                return;
            }
            ++depth;
            continue;
        }

        // Tail-recurse on the largest partition to strictly bound call stack depth to O(log N)
        if (effective_mid_size >= left_size && effective_mid_size >= right_size) {
            if (!left_slice.empty()) {
                multikey_quicksort_impl(left_slice, depth);
            }
            if (!right_slice.empty()) {
                multikey_quicksort_impl(right_slice, depth);
            }
            strings = mid_slice;
            ++depth;
            continue;
        }

        if (left_size >= right_size) {
            if (!right_slice.empty()) {
                multikey_quicksort_impl(right_slice, depth);
            }
            if (effective_mid_size > 0) {
                multikey_quicksort_impl(mid_slice, depth + 1);
            }
            strings = left_slice;
            continue;
        }

        if (!left_slice.empty()) {
            multikey_quicksort_impl(left_slice, depth);
        }
        if (effective_mid_size > 0) {
            multikey_quicksort_impl(mid_slice, depth + 1);
        }
        strings = right_slice;
    }

    if (strings.size() > 1) {
        string_insertion_sort(strings, depth);
    }
}

} // namespace detail

/**
 * @brief Sorts a span of strings in-place using Multi-Key Quicksort (Bentley & Sedgewick 1997).
 *
 * Operates with O(1) auxiliary heap memory and O(log N + max_len) stack depth.
 *
 * @tparam StringType String element type (default: std::string_view).
 * @param strings Contiguous span of string-like elements to sort in-place.
 */
template <typename StringType = std::string_view>
void multikey_quicksort(std::span<StringType> strings) {
    if (strings.size() <= 1) {
        return;
    }
    detail::multikey_quicksort_impl(strings, 0);
}

/**
 * @brief Sorts a contiguous random-access range of strings in-place using Multi-Key Quicksort.
 *
 * @tparam R Random-access range type with string elements.
 * @param range Range to sort in-place.
 */
template <std::ranges::random_access_range R>
    requires(!detail::is_span_v<std::remove_cvref_t<R>>)
void multikey_quicksort(R&& range) {
    if (std::ranges::size(range) <= 1) {
        return;
    }
    using ValueType = std::ranges::range_value_t<R>;
    if constexpr (std::ranges::contiguous_range<R>) {
        multikey_quicksort(
            std::span<ValueType>(std::ranges::data(range), std::ranges::size(range)));
    } else {
        // Fallback for non-contiguous random access ranges
        std::vector<ValueType> copy(std::ranges::begin(range), std::ranges::end(range));
        multikey_quicksort(std::span<ValueType>(copy));
        std::ranges::copy(copy, std::ranges::begin(range));
    }
}

} // namespace algoat::strings
