#pragma once

#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <iterator>
#include <ranges>

namespace algoat::selection::testing {

/**
 * @brief Verifies that the order statistic at nth satisfies the fundamental selection invariant:
 * for all elements before nth: !(comp(*nth, *it))
 * for all elements after nth:  !(comp(*it, *nth))
 */
template <typename Range, typename Iter, typename Comp = std::ranges::less,
          typename Proj = std::identity>
void verify_selection_invariant(Range&& r, Iter nth, Comp comp = {}, Proj proj = {}) {
    auto first = std::ranges::begin(r);
    auto last = std::ranges::end(r);
    if (first == last || nth == last) {
        return;
    }

    auto nth_val = std::invoke(proj, *nth);

    for (auto it = first; it != nth; ++it) {
        auto val = std::invoke(proj, *it);
        EXPECT_FALSE(std::invoke(comp, nth_val, val))
            << "Element before nth is greater than nth: " << val << " vs " << nth_val;
    }

    for (auto it = nth + 1; it != last; ++it) {
        auto val = std::invoke(proj, *it);
        EXPECT_FALSE(std::invoke(comp, val, nth_val))
            << "Element after nth is less than nth: " << val << " vs " << nth_val;
    }
}

/**
 * @brief Verifies that a multiple selection satisfies the multi-selection bucket invariant:
 * 1. For all elements before ranks[0]: !(comp(ranks[0], it))
 * 2. Between adjacent ranks (r_i, r_{i+1}):
 *    !(comp(it, ranks[r_i])) and !(comp(ranks[r_{i+1}], it))
 * 3. For all elements after ranks.back(): !(comp(it, ranks.back()))
 */
template <typename Range, typename RanksRange, typename Comp = std::ranges::less,
          typename Proj = std::identity>
void verify_multi_select_invariants(Range&& r, const RanksRange& ranks, Comp comp = {},
                                    Proj proj = {}) {
    auto first = std::ranges::begin(r);
    auto last = std::ranges::end(r);
    const auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
    const auto m = static_cast<std::size_t>(std::ranges::distance(ranks));

    if (n == 0 || m == 0) {
        return;
    }

    const auto rank_0 = static_cast<std::size_t>(ranks[0]);
    auto rank_0_val = std::invoke(proj, *(first + rank_0));

    // 1. Invariant for elements before ranks[0]
    for (std::size_t i = 0; i < rank_0; ++i) {
        auto val = std::invoke(proj, *(first + i));
        EXPECT_FALSE(std::invoke(comp, rank_0_val, val))
            << "Element at " << i << " > rank[0] at " << rank_0;
    }

    // 2. Invariant between consecutive ranks
    for (std::size_t r_idx = 0; r_idx + 1 < m; ++r_idx) {
        const auto cur_rank = static_cast<std::size_t>(ranks[r_idx]);
        const auto next_rank = static_cast<std::size_t>(ranks[r_idx + 1]);
        auto cur_val = std::invoke(proj, *(first + cur_rank));
        auto next_val = std::invoke(proj, *(first + next_rank));

        EXPECT_FALSE(std::invoke(comp, next_val, cur_val))
            << "Rank order violated between " << cur_rank << " and " << next_rank;

        for (std::size_t i = cur_rank + 1; i < next_rank; ++i) {
            auto val = std::invoke(proj, *(first + i));
            EXPECT_FALSE(std::invoke(comp, val, cur_val))
                << "Element at " << i << " < rank " << cur_rank;
            EXPECT_FALSE(std::invoke(comp, next_val, val))
                << "Element at " << i << " > rank " << next_rank;
        }
    }

    // 3. Invariant for elements after ranks.back()
    const auto last_rank = static_cast<std::size_t>(ranks[m - 1]);
    auto last_rank_val = std::invoke(proj, *(first + last_rank));
    for (std::size_t i = last_rank + 1; i < n; ++i) {
        auto val = std::invoke(proj, *(first + i));
        EXPECT_FALSE(std::invoke(comp, val, last_rank_val))
            << "Element at " << i << " < rank[last] at " << last_rank;
    }
}

} // namespace algoat::selection::testing
