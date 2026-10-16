/**
 * @file galloping_search.hpp
 * @brief Galloping Search (Exponential / Doubling Search) implementation.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/searching/branchless_binary_search.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::searching {

/**
 * @brief Galloping (exponential) search algorithm with O(log k) comparison bound.
 *
 * Implements Algorithm A1 from Bentley & Yao (1976).
 * In Phase 1, exponentially probes indices 1, 2, 4, 8, ... to bound the target in O(log k) steps.
 * In Phase 2, performs branchless binary search within the bounded window.
 *
 * Outperforms standard binary search when target rank k <= sqrt(N).
 *
 * @tparam I Random access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam T Target value type.
 * @tparam Proj Projection callable.
 * @tparam Comp Strict weak ordering comparator callable.
 *
 * @param first Beginning of range.
 * @param last End / sentinel of range.
 * @param target Value to locate lower bound for.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator pointing to the first element not less than target, or @c last if all elements
 * are less than target.
 */
namespace detail {

/**
 * @brief Core galloping search algorithm operating on any random-access iterator.
 */
template <typename I, typename S, typename T, typename Comp, typename Proj>
[[nodiscard]] constexpr I galloping_search_core(I first, S last, const T& target, Comp comp,
                                                Proj proj) {
    using diff_t = std::iter_difference_t<I>;
    const diff_t n = last - first;

    // Edge Case 1: Empty range
    if (n <= 0) {
        return first;
    }

    // Origin Fast Path: Target <= first element (Cost: Exactly 1 comparison)
    if (!bool(std::invoke(comp, std::invoke(proj, *first), target))) {
        return first;
    }

    // Phase 1: Exponential bounding (1, 2, 4, 8, 16, ...)
    // Invariant: all elements in [first, first + prev] are strictly < target.
    diff_t prev = 0;
    diff_t step = 1;

    while (step < n && bool(std::invoke(comp, std::invoke(proj, *(first + step)), target))) {
        prev = step;
        if (step > (n - 1) / 2) {
            step = n;
            break;
        }
        step <<= 1;
    }

    // Phase 2: Bounded Branchless Binary Search
    const diff_t sub_first = prev + 1;
    const diff_t sub_last = (step < n) ? step : n;

    return branchless_lower_bound(first + sub_first, first + sub_last, target, comp, proj);
}

} // namespace detail

template <
    std::random_access_iterator I, std::sentinel_for<I> S, typename T,
    typename Proj = std::identity,
    std::indirect_strict_weak_order<const T*, std::projected<I, Proj>> Comp = std::ranges::less>
[[nodiscard]] constexpr I galloping_search(I first, S last, const T& target, Comp comp = {},
                                           Proj proj = {}) {
    if constexpr (std::contiguous_iterator<I> && std::contiguous_iterator<S>) {
        if (!std::is_constant_evaluated()) {
            const auto* first_ptr = std::to_address(first);
            const auto* last_ptr = std::to_address(last);
            const auto* res_ptr =
                detail::galloping_search_core(first_ptr, last_ptr, target, comp, proj);
            return first + (res_ptr - first_ptr);
        }
    }

    return detail::galloping_search_core(first, last, target, comp, proj);
}

/**
 * @brief Overload of @c galloping_search accepting a C++20 random-access range.
 *
 * @tparam R Random-access range type.
 * @tparam T Target value type.
 * @tparam Proj Projection callable.
 * @tparam Comp Strict weak ordering comparator callable.
 *
 * @param range The sorted range to search.
 * @param target Target value.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Borrowed iterator pointing to the lower bound, or @c std::ranges::end(range).
 */
template <
    std::ranges::random_access_range R, typename T, typename Proj = std::identity,
    std::indirect_strict_weak_order<const T*, std::projected<std::ranges::iterator_t<R>, Proj>>
        Comp = std::ranges::less>
[[nodiscard]] constexpr std::ranges::borrowed_iterator_t<R>
galloping_search(R&& range, const T& target, Comp comp = {}, Proj proj = {}) {
    return galloping_search(std::ranges::begin(range), std::ranges::end(range), target,
                            std::move(comp), std::move(proj));
}

/**
 * @struct GallopingSearch
 * @brief Search algorithm adapter wrapping @c galloping_search to satisfy @c SearchAlgorithm.
 */
struct GallopingSearch {
    /**
     * @brief Returns unique algorithm identifier.
     * @return "gallopingsearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "gallopingsearch";
    }

    /**
     * @brief Indicates whether sorted input is required.
     * @return @c true
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }

    /**
     * @brief Searches for target within a const span using galloping search.
     *
     * @tparam T Element type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T>
    constexpr std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        return search(data, target, std::ranges::less{});
    }

    /**
     * @brief Searches for target within a mutable span using galloping search.
     *
     * @tparam T Element type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T>
    constexpr std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target, std::ranges::less{});
    }

    /**
     * @brief Searches for target within a const span using a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @param comp Strict weak ordering comparator.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T, typename Compare>
    constexpr std::optional<std::size_t> search(std::span<const T> data, const T& target,
                                                Compare comp) const {
        if (data.empty()) {
            return std::nullopt;
        }
        auto it = galloping_search(data.begin(), data.end(), target, comp);
        if (it != data.end() && !comp(*it, target) && !comp(target, *it)) {
            return static_cast<std::size_t>(it - data.begin());
        }
        return std::nullopt;
    }

    /**
     * @brief Searches for target within a mutable span using a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @param comp Strict weak ordering comparator.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T, typename Compare>
    constexpr std::optional<std::size_t> search(std::span<T> data, const T& target,
                                                Compare comp) const {
        return search(std::span<const T>{data.data(), data.size()}, target, std::move(comp));
    }
};

} // namespace algoat::searching

ALGOAT_REGISTER_ALGORITHM("searching", "gallopingsearch", ::algoat::searching::GallopingSearch)
