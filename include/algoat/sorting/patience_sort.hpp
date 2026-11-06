/**
 * @file patience_sort.hpp
 * @brief Patience Sorting with Simultaneous Longest Increasing Subsequence (LIS) Extraction.
 *
 * Implements card-solitaire patience sorting based on Aldous & Diaconis (1999) and
 * Mallows (1973). Finds the leftmost legal pile via branchless binary search, tracks
 * predecessor back-pointers to reconstruct the exact LIS in O(LIS) steps, and merges
 * piles into sorted order using a tournament LoserTree in O(N log K) time.
 *
 * Characteristics:
 * - Time Complexity: O(N log K) comparisons (where K = LIS length <= N)
 *   - Best-case (reverse sorted, K = 1): O(N) comparisons
 *   - Average-case (random permutation, K ~ 2 sqrt(N)): O(N log N)
 *   - Worst-case (sorted, K = N): O(N log N)
 * - Space Complexity: O(N) auxiliary space for piles and back-pointer index arrays
 * - Capabilities: Produces both fully sorted array and exact LIS indices simultaneously
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/sorting/loser_tree.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace algoat::sorting {

/**
 * @struct PatienceSortResult
 * @brief Return payload containing both sorted sequence and exact LIS indices.
 *
 * @tparam T Value type of the sorted elements.
 */
template <typename T> struct PatienceSortResult {
    std::vector<T> sorted_data;
    std::vector<std::size_t> lis_indices; // Indices in original range forming the LIS
};

/**
 * @brief Sorts the iterator range [first, last) using Patience Sort and extracts LIS.
 *
 * @tparam I Forward iterator type.
 * @tparam S Sentinel type.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable.
 * @param first Beginning of range.
 * @param last Sentinel / end of range.
 * @param comp Strict weak ordering comparator (default: std::ranges::less).
 * @param proj Element projection callable (default: std::identity).
 * @return PatienceSortResult containing sorted elements and LIS indices.
 */
template <std::forward_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::indirect_strict_weak_order<Comp, std::projected<I, Proj>>
PatienceSortResult<std::iter_value_t<I>> patience_sort(I first, S last, Comp comp = {},
                                                       Proj proj = {}) {
    using T = std::iter_value_t<I>;
    PatienceSortResult<T> result;

    if (first == last) {
        return result;
    }

    // Piles: Each pile is stored with bottom at index 0 and top at back().
    // By patience sorting rules, bottom >= ... >= top.
    // Popping from back() yields elements in ascending order.
    std::vector<std::vector<T>> piles;
    std::vector<T> pile_tops;
    std::vector<std::size_t> pile_top_indices;
    std::vector<std::size_t> pred;

    constexpr std::size_t kSentinel = static_cast<std::size_t>(-1);

    std::size_t input_idx = 0;
    for (auto it = first; it != last; ++it, ++input_idx) {
        T val = *it;
        auto val_projected = std::invoke(proj, val);

        // Find the leftmost pile j whose top is >= val_projected using branchless search
        auto top_comp = [&](const T& top_elem, const auto& target_proj) {
            return bool(std::invoke(comp, std::invoke(proj, top_elem), target_proj));
        };

        auto bound_it = algoat::searching::branchless_lower_bound(
            pile_tops.begin(), pile_tops.end(), val_projected, top_comp);
        const std::size_t pile_idx = static_cast<std::size_t>(bound_it - pile_tops.begin());

        pred.push_back(pile_idx > 0 ? pile_top_indices[pile_idx - 1] : kSentinel);

        if (pile_idx == piles.size()) {
            // New pile created
            pile_tops.push_back(val);
            pile_top_indices.push_back(input_idx);
            piles.push_back(std::vector<T>{std::move(val)});
        } else {
            // Place on existing pile
            pile_tops[pile_idx] = val;
            pile_top_indices[pile_idx] = input_idx;
            piles[pile_idx].push_back(std::move(val));
        }
    }

    const std::size_t num_piles = piles.size();

    // 1. Reconstruct Longest Increasing Subsequence (LIS)
    if (num_piles > 0) {
        result.lis_indices.reserve(num_piles);
        std::size_t curr = pile_top_indices[num_piles - 1];
        while (curr != kSentinel) {
            result.lis_indices.push_back(curr);
            curr = pred[curr];
        }
        std::ranges::reverse(result.lis_indices);
    }

    // 2. Multi-way Stream Merge into sorted_data using LoserTree
    result.sorted_data.reserve(input_idx);
    if (num_piles == 1) {
        // Fast-path: Single pile popped top-to-bottom is already ascending
        auto& single_pile = piles[0];
        while (!single_pile.empty()) {
            result.sorted_data.push_back(std::move(single_pile.back()));
            single_pile.pop_back();
        }
    } else if (num_piles > 1) {
        auto item_comp = [&](const T& a, const T& b) {
            return bool(std::invoke(comp, std::invoke(proj, a), std::invoke(proj, b)));
        };

        std::vector<T> stream_heads;
        stream_heads.reserve(num_piles);
        for (const auto& p : piles) {
            stream_heads.push_back(p.back());
        }

        algoat::sorting::LoserTree<T, decltype(item_comp)> tree(num_piles, item_comp);
        tree.initialize(stream_heads);

        while (!tree.empty()) {
            const std::size_t winner = tree.top_source();
            result.sorted_data.push_back(tree.top());
            piles[winner].pop_back();
            if (!piles[winner].empty()) {
                tree.replace_top(piles[winner].back());
            } else {
                tree.pop_stream();
            }
        }
    }

    return result;
}

/**
 * @brief Sorts a forward range using Patience Sort and extracts LIS.
 *
 * @tparam R Forward range type.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable.
 * @param data Range to sort.
 * @param comp Strict weak ordering comparator (default: std::ranges::less).
 * @param proj Element projection callable (default: std::identity).
 * @return PatienceSortResult containing sorted sequence and LIS indices.
 */
template <std::ranges::forward_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::indirect_strict_weak_order<Comp, std::projected<std::ranges::iterator_t<R>, Proj>>
PatienceSortResult<std::ranges::range_value_t<R>> patience_sort(R&& data, Comp comp = {},
                                                                Proj proj = {}) {
    return patience_sort(std::ranges::begin(data), std::ranges::end(data), std::move(comp),
                         std::move(proj));
}

/**
 * @brief Sorts a contiguous span of elements using Patience Sort.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 * @param data Span of elements.
 * @param comp Strict weak ordering comparator.
 * @return PatienceSortResult containing sorted data and LIS indices.
 */
template <typename T, typename Compare = std::ranges::less>
PatienceSortResult<T> patience_sort(std::span<const T> data, Compare comp = {}) {
    return patience_sort(data.begin(), data.end(), std::move(comp), std::identity{});
}

template <typename T, typename Compare = std::ranges::less>
PatienceSortResult<T> patience_sort(std::span<T> data, Compare comp = {}) {
    return patience_sort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @struct PatienceSort
 * @brief Zero-overhead adapter struct for PatienceSort satisfying SortAlgorithm concept.
 */
struct PatienceSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "patiencesort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    template <typename T> void sort(std::span<T> data) const {
        auto res = patience_sort(data);
        std::ranges::copy(res.sorted_data, data.begin());
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        auto res = patience_sort(data, std::move(comp));
        std::ranges::copy(res.sorted_data, data.begin());
    }
};

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "patiencesort", ::algoat::sorting::PatienceSort)
