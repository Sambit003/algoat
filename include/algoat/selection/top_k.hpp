/**
 * @file top_k.hpp
 * @brief In-place range selection operations (top_k_unsorted and top_k_sorted).
 */

#pragma once

#include "algoat/selection/select_optimal.hpp"
#include "algoat/sorting/pdqsort.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <utility>

namespace algoat::selection {

/**
 * @brief Selects the k smallest elements of range @c r in O(N) expected time.
 *
 * After execution, the prefix subrange [0, k) contains the k smallest elements
 * in unspecified order, and all elements in [0, k) are less than or equal to
 * all elements in [k, n).
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param r Range of elements to rearrange.
 * @param k Number of smallest elements to select into prefix.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
void top_k_unsorted(R&& r, std::size_t k, Comp comp = {}, Proj proj = {}) {
    auto first = std::ranges::begin(r);
    auto last = std::ranges::end(r);
    auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
    if (k == 0 || k >= n) {
        return;
    }
    select_optimal(first, first + k, last, std::move(comp), std::move(proj));
}

/**
 * @brief Selects and sorts the k smallest elements of range @c r in O(N + K log K) time.
 *
 * After execution, the prefix subrange [0, k) contains the k smallest elements
 * in strictly sorted ascending order according to @c comp and @c proj.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param r Range of elements to rearrange.
 * @param k Number of smallest elements to select and sort into prefix.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
void top_k_sorted(R&& r, std::size_t k, Comp comp = {}, Proj proj = {}) {
    auto first = std::ranges::begin(r);
    auto last = std::ranges::end(r);
    auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
    if (k == 0) {
        return;
    }
    if (k >= n) {
        sorting::pdqsort(first, last, comp, proj);
        return;
    }
    select_optimal(first, first + k, last, comp, proj);
    sorting::pdqsort(first, first + k, std::move(comp), std::move(proj));
}

} // namespace algoat::selection
