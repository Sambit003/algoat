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

} // namespace algoat::selection::testing
