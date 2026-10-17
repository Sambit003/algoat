/**
 * @file nth_element.hpp
 * @brief Musser's Introspective Selection (Introselect) with guaranteed worst-case O(N) bound.
 */

#pragma once

#include "algoat/selection/detail/bfprt.hpp"
#include "algoat/sorting/insertionsort.hpp"

#include <bit>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <utility>

namespace algoat::selection {

namespace detail {

/**
 * @brief Selects the median of three elements according to comparator and projection.
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param a First element iterator.
 * @param b Second element iterator.
 * @param c Third element iterator.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator pointing to the median among a, b, and c.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
    requires std::sortable<I, Comp, Proj>
constexpr I median_of_3(I a, I b, I c, Comp comp, Proj proj) {
    if (std::invoke(comp, std::invoke(proj, *b), std::invoke(proj, *a))) {
        std::ranges::iter_swap(a, b);
    }
    if (std::invoke(comp, std::invoke(proj, *c), std::invoke(proj, *b))) {
        std::ranges::iter_swap(b, c);
        if (std::invoke(comp, std::invoke(proj, *b), std::invoke(proj, *a))) {
            std::ranges::iter_swap(a, b);
        }
    }
    return b;
}

/**
 * @brief Core Introselect loop bounding recursion depth and falling back to BFPRT.
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range.
 * @param nth Target order statistic iterator.
 * @param last Sentinel or end of the range.
 * @param depth_limit Maximum remaining partition depth before triggering BFPRT.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
    requires std::sortable<I, Comp, Proj>
constexpr void introselect_loop(I first, I nth, I last, int depth_limit, Comp comp, Proj proj) {
    while (last - first > 24) {
        if (depth_limit == 0) {
            // Depleted depth budget: abort Quickselect and trigger BFPRT Median-of-Medians
            bfprt_select(first, nth, last, comp, proj);
            return;
        }
        --depth_limit;

        I mid = first + (last - first) / 2;
        I pivot_it = median_of_3(first, mid, last - 1, comp, proj);
        auto pivot = std::invoke(proj, *pivot_it);

        auto [mid_first, mid_last] = partition_3way(first, last, pivot, comp, proj);

        if (nth < mid_first) {
            last = mid_first;
        } else if (nth >= mid_last) {
            first = mid_last;
        } else {
            return; // nth is already positioned within the duplicate pivot run
        }
    }
    sorting::insertion_sort(first, last, comp, proj);
}

} // namespace detail

/**
 * @brief Rearranges elements such that the element at @c nth is the one that would be in that
 * position if [first, last) were sorted.
 *
 * Implements Musser's Introspective Selection with guaranteed O(N) worst-case time by monitoring
 * partitioning depth and falling back to Blum-Floyd-Pratt-Rivest-Tarjan (BFPRT) Median-of-Medians.
 *
 * @tparam I Random-access iterator type.
 * @tparam S Sentinel type for I.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range to rearrange.
 * @param nth Iterator pointing to the order statistic to select.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator pointing to the end of the range.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr I nth_element(I first, I nth, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    if (first == last_it || nth == last_it) {
        return last_it;
    }
    auto len = last_it - first;
    if (len <= 1) {
        return last_it;
    }

    int depth_limit = 2 * (std::bit_width(static_cast<std::size_t>(len)) - 1);
    detail::introselect_loop(first, nth, last_it, depth_limit, comp, proj);
    return last_it;
}

/**
 * @brief Range overload for @c algoat::selection::nth_element.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param r Random-access range to rearrange.
 * @param nth Iterator pointing to the order statistic to select.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> nth_element(R&& r, std::ranges::iterator_t<R> nth,
                                                          Comp comp = {}, Proj proj = {}) {
    return nth_element(std::ranges::begin(r), nth, std::ranges::end(r), std::move(comp),
                       std::move(proj));
}

} // namespace algoat::selection
