#include "algoat/searching/veb_tree.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <set>
#include <vector>

using namespace algoat::searching;

RC_GTEST_PROP(VEBTreePBT, SetEquivalence,
              (const std::vector<uint16_t>& inserts, const std::vector<uint16_t>& erases)) {
    VanEmdeBoasTree<16> veb;
    std::set<uint32_t> ref_set;

    for (uint16_t x : inserts) {
        veb.insert(x);
        ref_set.insert(x);
    }

    for (uint16_t x : erases) {
        veb.erase(x);
        ref_set.erase(x);
    }

    RC_ASSERT(veb.empty() == ref_set.empty());

    if (ref_set.empty()) {
        RC_ASSERT(!veb.min().has_value());
        RC_ASSERT(!veb.max().has_value());
    } else {
        RC_ASSERT(veb.min().has_value());
        RC_ASSERT(veb.max().has_value());
        RC_ASSERT(*veb.min() == *ref_set.begin());
        RC_ASSERT(*veb.max() == *ref_set.rbegin());
        RC_ASSERT(*veb.min() <= *veb.max());
        RC_ASSERT(veb.contains(*veb.min()));
        RC_ASSERT(veb.contains(*veb.max()));
    }

    // Verify sample queries
    for (uint16_t x : inserts) {
        bool expected = (ref_set.count(x) > 0);
        RC_ASSERT(veb.contains(x) == expected);

        auto it = ref_set.upper_bound(x);
        std::optional<uint32_t> expected_succ =
            (it != ref_set.end()) ? std::make_optional(*it) : std::nullopt;
        RC_ASSERT(veb.successor(x) == expected_succ);

        auto it_pred = ref_set.lower_bound(x);
        std::optional<uint32_t> expected_pred = std::nullopt;
        if (it_pred != ref_set.begin()) {
            --it_pred;
            expected_pred = *it_pred;
        }
        RC_ASSERT(veb.predecessor(x) == expected_pred);
    }
}

RC_GTEST_PROP(VEBTreePBT, ClearInvariants, (const std::vector<uint16_t>& vals)) {
    VanEmdeBoasTree<16> veb;
    for (uint16_t x : vals) {
        veb.insert(x);
    }

    veb.clear();
    RC_ASSERT(veb.empty());
    RC_ASSERT(!veb.min().has_value());
    RC_ASSERT(!veb.max().has_value());

    for (uint16_t x : vals) {
        RC_ASSERT(!veb.contains(x));
    }
}
