#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace algoat::strings {

namespace detail {

/**
 * @brief Computes the critical factorization and fundamental period of needle.
 *
 * Implements the Crochemore-Perrin (1991) critical factorization algorithm using
 * Duval's maximal suffix algorithm across two total orders (<= and >=).
 * Guarantees O(M) time and strictly O(1) auxiliary space (no dynamic allocation).
 *
 * @param needle Pattern string view.
 * @param[out] period Fundamental period of the maximal suffix.
 * @return std::size_t The critical factorization cut index ell (|u| = ell).
 */
[[nodiscard]] inline std::size_t critical_factorization(std::string_view needle,
                                                        std::size_t& period) noexcept {
    const std::size_t m = needle.size();

    // Helper evaluating Duval's maximal suffix for a given strict order
    auto compute_max_suffix = [&](auto comp) noexcept -> std::pair<std::size_t, std::size_t> {
        std::size_t ms = static_cast<std::size_t>(-1);
        std::size_t j = 0;
        std::size_t k = 1;
        std::size_t p = 1;
        while (j + k < m) {
            const auto a = static_cast<unsigned char>(needle[j + k]);
            const auto b = static_cast<unsigned char>(needle[ms + k]);
            if (comp(a, b)) {
                j += k;
                k = 1;
                p = j - ms;
            } else if (a == b) {
                if (k != p) {
                    ++k;
                } else {
                    j += p;
                    k = 1;
                }
            } else {
                ms = j++;
                k = p = 1;
            }
        }
        return {ms + 1, p};
    };

    const auto [ms1, p1] = compute_max_suffix(std::less<unsigned char>{});
    const auto [ms2, p2] = compute_max_suffix(std::greater<unsigned char>{});

    if (ms1 >= ms2) {
        period = p1;
        return ms1;
    }
    period = p2;
    return ms2;
}

/**
 * @brief Crochemore-Perrin (1991) Two-Way exact substring search.
 *
 * Guarantees at most 2N - M character comparisons and strictly O(1) auxiliary space.
 * Prevents quadratic degradation on repetitive / pathological patterns.
 *
 * @param haystack Text to search in.
 * @param needle Pattern to find.
 * @return std::string_view::size_type Index of first occurrence, or std::string_view::npos.
 */
[[nodiscard]] inline std::string_view::size_type two_way_search(std::string_view haystack,
                                                                std::string_view needle) noexcept {
    const std::size_t n = haystack.size();
    const std::size_t m = needle.size();

    if (m == 0) {
        return 0;
    }
    if (n < m) {
        return std::string_view::npos;
    }
    if (m == 1) {
        const auto* ptr = static_cast<const char*>(std::memchr(haystack.data(), needle[0], n));
        return ptr ? static_cast<std::string_view::size_type>(ptr - haystack.data())
                   : std::string_view::npos;
    }

    std::size_t period = 0;
    const std::size_t ell = critical_factorization(needle, period);

    const bool is_periodic =
        (period + ell <= m) && (std::memcmp(needle.data(), needle.data() + period, ell) == 0);

    if (is_periodic) {
        std::size_t memory = 0;
        std::size_t j = 0;
        while (j + m <= n) {
            // Scan right factor v
            std::size_t i = std::max(ell, memory);
            while (i < m && needle[i] == haystack[i + j]) {
                ++i;
            }
            if (i >= m) {
                // Scan left factor u
                i = ell;
                while (i > memory && needle[i - 1] == haystack[i - 1 + j]) {
                    --i;
                }
                if (i <= memory) {
                    return j;
                }
                j += period;
                memory = m - period;
            } else {
                j += (i - ell + 1);
                memory = 0;
            }
        }
    } else {
        const std::size_t non_periodic_period = std::max(ell, m - ell) + 1;
        std::size_t j = 0;
        while (j + m <= n) {
            // Scan right factor v
            std::size_t i = ell;
            while (i < m && needle[i] == haystack[i + j]) {
                ++i;
            }
            if (i >= m) {
                // Scan left factor u
                i = ell;
                while (i > 0 && needle[i - 1] == haystack[i - 1 + j]) {
                    --i;
                }
                if (i == 0) {
                    return j;
                }
                j += non_periodic_period;
            } else {
                j += (i - ell + 1);
            }
        }
    }

    return std::string_view::npos;
}

#if !defined(ALGOAT_SUBSTRING_SEARCH_SCALAR_ONLY)
/**
 * @brief Forward declaration of the Highway dynamic multi-target SIMD dispatch engine
 *        compiled in src/strings/substring_search.cpp across AVX-512, AVX2, SSE4, NEON.
 */
std::string_view::size_type find_substring_simd_dispatch(const char* hs_data, std::size_t n,
                                                         const char* ne_data,
                                                         std::size_t m) noexcept;
#endif

template <typename T> [[nodiscard]] inline std::string_view as_string_view(const T& val) noexcept {
    if constexpr (std::is_convertible_v<const T&, std::string_view>) {
        return std::string_view(val);
    } else {
        return std::string_view(reinterpret_cast<const char*>(std::ranges::data(val)),
                                std::ranges::size(val));
    }
}

} // namespace detail

/**
 * @brief Finds the first occurrence of needle in haystack using Highway SIMD pre-filtering
 *        and Crochemore-Perrin Two-Way verification.
 *
 * @param haystack String view of the text to search in.
 * @param needle String view of the pattern to search for.
 * @return std::string_view::size_type Index of first occurrence, or std::string_view::npos.
 */
[[nodiscard]] inline std::string_view::size_type find_substring(std::string_view haystack,
                                                                std::string_view needle) noexcept {
#if !defined(ALGOAT_SUBSTRING_SEARCH_SCALAR_ONLY)
    return detail::find_substring_simd_dispatch(haystack.data(), haystack.size(), needle.data(),
                                                needle.size());
#else
    return detail::two_way_search(haystack, needle);
#endif
}

/**
 * @brief Finds all occurrences of needle in haystack.
 *
 * @param haystack String view of the text to search in.
 * @param needle String view of the pattern to search for.
 * @return std::vector<std::string_view::size_type> Starting indices of all matches.
 */
[[nodiscard]] inline std::vector<std::string_view::size_type>
find_all_substrings(std::string_view haystack, std::string_view needle) {
    std::vector<std::string_view::size_type> matches;
    if (needle.empty() || haystack.size() < needle.size()) {
        return matches;
    }

    std::string_view::size_type offset = 0;
    while (offset <= haystack.size() - needle.size()) {
        const auto pos = find_substring(haystack.substr(offset), needle);
        if (pos == std::string_view::npos) {
            break;
        }
        const auto match_pos = offset + pos;
        matches.push_back(match_pos);
        offset = match_pos + 1;
    }
    return matches;
}

/**
 * @brief Range-based C++20 search adaptor.
 *
 * Specializes for contiguous byte/char ranges to use the SIMD-accelerated substring search engine,
 * while seamlessly delegating generic forward ranges to std::ranges::search.
 * Correctly respects the std::ranges::borrowed_range contract for rvalues.
 *
 * @tparam R1 Type satisfying std::ranges::forward_range.
 * @tparam R2 Type satisfying std::ranges::forward_range.
 * @param haystack Range to search within.
 * @param needle Pattern range to locate.
 * @return std::ranges::borrowed_iterator_t<R1> Iterator pointing to match start, or end iterator.
 */
template <std::ranges::forward_range R1, std::ranges::forward_range R2>
[[nodiscard]] inline std::ranges::borrowed_iterator_t<R1> search_pattern(R1&& haystack,
                                                                         R2&& needle) {
    using Char1 = std::ranges::range_value_t<R1>;
    using Char2 = std::ranges::range_value_t<R2>;

    if constexpr (!std::ranges::borrowed_range<R1>) {
        return std::ranges::dangling{};
    } else if constexpr (std::ranges::contiguous_range<R1> && std::ranges::contiguous_range<R2> &&
                         (sizeof(Char1) == 1) && (sizeof(Char2) == 1)) {
        const std::string_view hs = detail::as_string_view(haystack);
        const std::string_view ne = detail::as_string_view(needle);
        const auto pos = find_substring(hs, ne);
        if (pos == std::string_view::npos) {
            return std::ranges::end(haystack);
        }
        return std::ranges::next(std::ranges::begin(haystack), pos);
    } else {
        auto sub = std::ranges::search(std::forward<R1>(haystack), std::forward<R2>(needle));
        return sub.begin();
    }
}

} // namespace algoat::strings
