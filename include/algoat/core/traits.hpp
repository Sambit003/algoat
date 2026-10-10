/**
 * @file traits.hpp
 * @brief Data profiling and traits analysis for runtime algorithm selection.
 *
 * Provides utilities to inspect input sequences in a single O(n) pass to measure
 * characteristics like size, sortedness ratio, and presence of duplicate elements.
 */

#pragma once

#include "algoat/core/simd_profiler.hpp"

#include <complex>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace algoat::detail {

/**
 * @brief Composes a comparator and a projection into a single binary predicate callable.
 */
template <typename Comp, typename Proj>
[[nodiscard]] constexpr auto make_comp_proj(Comp comp, Proj proj) {
    return [comp = std::move(comp), proj = std::move(proj)](auto&& a, auto&& b) -> bool {
        return std::invoke(comp, std::invoke(proj, std::forward<decltype(a)>(a)),
                           std::invoke(proj, std::forward<decltype(b)>(b)));
    };
}

/**
 * @brief Concept matching ranges that are both contiguous and sized.
 */
template <typename R>
concept contiguous_sized_range = std::ranges::contiguous_range<R> && std::ranges::sized_range<R>;

/**
 * @brief Zero-overhead adapter converting any contiguous sized range to std::span<T>.
 */
template <contiguous_sized_range R> [[nodiscard]] constexpr auto to_span(R&& r) noexcept {
    using Element = std::remove_reference_t<std::ranges::range_reference_t<R>>;
    return std::span<Element>{std::ranges::data(r), std::ranges::size(r)};
}

/**
 * @brief Zero-allocation fallback sort for non-contiguous random-access ranges.
 */
template <typename R, typename Comp, typename Proj>
inline auto fallback_sort(R&& data, Comp comp, Proj proj) {
    auto comp_proj = make_comp_proj(std::move(comp), std::move(proj));
    std::ranges::sort(data, comp_proj);
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

/**
 * @brief Zero-allocation fallback stable sort for non-contiguous random-access ranges.
 */
template <typename R, typename Comp, typename Proj>
inline auto fallback_stable_sort(R&& data, Comp comp, Proj proj) {
    auto comp_proj = make_comp_proj(std::move(comp), std::move(proj));
    std::ranges::stable_sort(data, comp_proj);
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

} // namespace algoat::detail

namespace algoat::core {

/**
 * @brief Type trait to identify std::complex types.
 * @tparam T The type to check.
 */
template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

/**
 * @brief Helper variable template for is_complex.
 */
template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

/**
 * @brief Concept matching std::complex instances.
 */
template <typename T>
concept IsComplex = is_complex_v<std::remove_cvref_t<T>>;

/**
 * @brief Concept matching boolean data types.
 */
template <typename T>
concept IsBoolean = std::same_as<std::remove_cvref_t<T>, bool>;

/**
 * @struct DataTraits
 * @brief Represents statistical and structural properties of a dataset.
 *
 * Used by @c algoat::core::Dispatcher to make intelligent heuristic decisions
 * regarding algorithm selection (e.g., choosing TimSort for nearly sorted arrays
 * or InsertionSort for small arrays).
 */
struct DataTraits {
    std::size_t size;        ///< Number of elements in the range.
    double sortedness_ratio; ///< Sortedness metric: 0.0 = completely reversed, 1.0 = fully sorted.
    bool has_duplicates;     ///< True if at least one adjacent duplicate pair was detected.
    bool is_exact{
        true}; ///< True if traits were calculated via exact scan; false if sampled via SIMD.
};

/**
 * @brief Analyzes a random access range to extract structural traits.
 *
 * For small sequences (N <= 10,000) or non-contiguous ranges, performs an exact O(N) pass.
 * For large contiguous sequences (N > 10,000), executes a sub-linear O(1) cache-line-aware
 * stratified SIMD profiler with high statistical confidence.
 *
 * @tparam R A type satisfying @c std::ranges::random_access_range.
 * @tparam Comp Comparator callable defining strict weak ordering.
 * @tparam Proj Projection callable.
 *
 * @param data The range of elements to analyze.
 * @param comp Strict weak ordering comparator.
 * @param proj Projection callable.
 * @return @c DataTraits Computed traits (@c size, @c sortedness_ratio, @c has_duplicates, @c
 * is_exact).
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
DataTraits analyze(const R& data, Comp comp = {}, Proj proj = {}) {
    const std::size_t size = std::ranges::size(data);
    if (size <= 1) {
        return {size, 1.0, false, true};
    }

    using ElementType = std::remove_cvref_t<std::ranges::range_value_t<R>>;

    // Sub-linear fast path for large contiguous primitive sequences with default comparator and
    // projection
    if constexpr (std::ranges::contiguous_range<R> &&
                  (std::is_arithmetic_v<ElementType> || std::is_pointer_v<ElementType>) &&
                  std::is_same_v<Comp, std::ranges::less> && std::is_same_v<Proj, std::identity>) {
        if (size > detail::kSublinearThreshold) {
            double ratio = 1.0;
            bool has_duplicates = false;
            detail::sample_traits_sublinear_impl(std::ranges::data(data), size, ratio,
                                                 has_duplicates);
            return {size, ratio, has_duplicates, /*is_exact=*/false};
        }
    }

    // Exact O(N) scalar pass for small arrays or non-contiguous/custom ranges
    std::size_t sorted_pairs = 0;
    bool has_duplicates = false;

    auto it = std::ranges::begin(data);
    auto prev = it;
    ++it;

    for (; it != std::ranges::end(data); ++it, ++prev) {
        auto&& prev_proj = std::invoke(proj, *prev);
        auto&& curr_proj = std::invoke(proj, *it);
        if (!std::invoke(comp, curr_proj, prev_proj)) {
            sorted_pairs++;
        }
        if (!std::invoke(comp, prev_proj, curr_proj) && !std::invoke(comp, curr_proj, prev_proj)) {
            has_duplicates = true;
        }
    }

    // A fully sorted array has (size - 1) sorted pairs.
    double ratio = static_cast<double>(sorted_pairs) / static_cast<double>(size - 1);

    return {size, ratio, has_duplicates, /*is_exact=*/true};
}

} // namespace algoat::core
