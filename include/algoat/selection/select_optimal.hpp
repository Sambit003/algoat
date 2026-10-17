/**
 * @file select_optimal.hpp
 * @brief Floyd-Rivest (1975 Algorithm 489) optimal selection algorithm.
 *
 * Approaches the theoretical Cunto-Munro lower bound (~1.5N comparisons for median).
 */

#pragma once

#include "algoat/selection/nth_element.hpp"
#include "algoat/sorting/insertionsort.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

namespace algoat::selection {

namespace detail {

/**
 * @brief In-place Floyd-Rivest Algorithm 489 selection loop.
 *
 * Employs recursive sublinear sampling of size S = 0.5 * N^(2/3) with standard deviation
 * directional bias, followed by sentinel-based Hoare partitioning without bounds checks.
 *
 * @tparam I Random-access iterator type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 *
 * @param first Beginning of the range.
 * @param nth Target order statistic iterator.
 * @param last Sentinel or end of the range.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
    requires std::sortable<I, Comp, Proj>
void floyd_rivest_loop(I first, I nth, I last, Comp comp, Proj proj) {
    std::ptrdiff_t range_left = 0;
    std::ptrdiff_t range_right = std::distance(first, last) - 1;
    const std::ptrdiff_t target_k = std::distance(first, nth);

    while (range_right > range_left) {
        if (range_right - range_left <= 24) {
            sorting::insertion_sort(first + range_left, first + range_right + 1, comp, proj);
            return;
        }

        // Sublinear recursive sampling when subarray length exceeds 600
        if (range_right - range_left > 600) {
            const double current_len = static_cast<double>(range_right - range_left + 1);
            const double rank_in_range = static_cast<double>(target_k - range_left + 1);
            const double log_n = std::log(current_len);
            const double sample_size = 0.5 * std::exp(2.0 * log_n / 3.0);

            // Theodore Brown (1976) continuous directional bias formulation
            const double sign_bias = (2.0 * rank_in_range / current_len) - 1.0;
            const double sd_bias =
                0.5 * std::sqrt(log_n * sample_size * (current_len - sample_size) / current_len) *
                (sign_bias < 0.0 ? -1.0 : (sign_bias > 0.0 ? 1.0 : 0.0));

            std::ptrdiff_t sample_first_idx =
                range_left +
                static_cast<std::ptrdiff_t>(std::round(
                    (rank_in_range - 1.0) - (rank_in_range * sample_size / current_len) + sd_bias));
            std::ptrdiff_t sample_last_idx =
                range_left +
                static_cast<std::ptrdiff_t>(std::round(
                    (rank_in_range - 1.0) +
                    ((current_len - rank_in_range) * sample_size / current_len) + sd_bias));

            // Defensive clamping within current range bounds
            sample_first_idx = std::clamp(sample_first_idx, range_left, target_k);
            sample_last_idx = std::clamp(sample_last_idx, target_k, range_right);

            // Recursive call to select pivot in sample
            floyd_rivest_loop(first + sample_first_idx, first + target_k,
                              first + sample_last_idx + 1, comp, proj);
        }

        // Partition range [first + range_left, first + range_right] around pivot
        auto pivot_val = std::invoke(proj, *(first + target_k));
        std::ptrdiff_t left_idx = range_left;
        std::ptrdiff_t right_idx = range_right;

        std::ranges::iter_swap(first + range_left, first + target_k);
        bool pivot_at_left = false;
        if (std::invoke(comp, pivot_val, std::invoke(proj, *(first + range_right)))) {
            std::ranges::iter_swap(first + range_right, first + range_left);
            pivot_at_left = true;
        }

        while (left_idx < right_idx) {
            std::ranges::iter_swap(first + left_idx, first + right_idx);
            ++left_idx;
            --right_idx;
            while (std::invoke(comp, std::invoke(proj, *(first + left_idx)), pivot_val)) {
                ++left_idx;
            }
            while (std::invoke(comp, pivot_val, std::invoke(proj, *(first + right_idx)))) {
                --right_idx;
            }
        }

        if (pivot_at_left) {
            std::ranges::iter_swap(first + range_left, first + right_idx);
        } else {
            ++right_idx;
            std::ranges::iter_swap(first + right_idx, first + range_right);
        }

        // Adjust search bounds to the partition containing target index k
        if (right_idx <= target_k) {
            range_left = right_idx + 1;
        }
        if (target_k <= right_idx) {
            range_right = right_idx - 1;
        }
    }
}

} // namespace detail

/**
 * @brief Rearranges elements in [first, last) such that the element at @c nth is in its final
 * sorted position, using Floyd-Rivest (1975 Algorithm 489) optimal sampling selection.
 *
 * Achieves an average comparison bound of N + min(K, N - K) + o(N) comparisons (approximately
 * 1.5N for the median), reaching the theoretical Cunto-Munro lower bound.
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
constexpr I select_optimal(I first, I nth, S last, Comp comp = {}, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    if (first == last_it || nth == last_it) {
        return last_it;
    }
    if (std::is_constant_evaluated()) {
        return nth_element(first, nth, last, comp, proj);
    }
    auto len = last_it - first;
    if (len <= 1) {
        return last_it;
    }

    detail::floyd_rivest_loop(first, nth, last_it, comp, proj);
    return last_it;
}

/**
 * @brief Range overload for @c algoat::selection::select_optimal.
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
constexpr std::ranges::borrowed_iterator_t<R> select_optimal(R&& r, std::ranges::iterator_t<R> nth,
                                                             Comp comp = {}, Proj proj = {}) {
    return select_optimal(std::ranges::begin(r), nth, std::ranges::end(r), std::move(comp),
                          std::move(proj));
}

} // namespace algoat::selection
