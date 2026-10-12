/**
 * @file test_module_invariants.cpp
 * @brief Property-based tests verifying algorithm invariants through import algoat;.
 */

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

import algoat;

RC_GTEST_PROP(ModulePBT, SortPreservesElementsAndOrders, (std::vector<int> original)) {
    auto sorted = original;
    algoat::sort(sorted);

    // 1. Invariant: Range is partitioned in non-decreasing order
    RC_ASSERT(std::is_sorted(sorted.begin(), sorted.end()));

    // 2. Invariant: Permutation preservation (multiset equivalence)
    RC_ASSERT(std::is_permutation(sorted.begin(), sorted.end(), original.begin(), original.end()));
}

RC_GTEST_PROP(ModulePBT, SearchFindsInsertedElementInSortedData,
              (std::vector<int> input, int target)) {
    input.push_back(target);
    algoat::sort(input);

    auto found = algoat::search(input, target);
    RC_ASSERT(found.has_value());
    RC_ASSERT(*found < input.size());
    RC_ASSERT(input[*found] == target);
}

RC_GTEST_PROP(ModulePBT, PdqSortPreservesElementsAndOrders, (std::vector<int> original)) {
    auto sorted = original;
    algoat::sorting::pdqsort(sorted);

    // Invariant 1: Sortedness
    RC_ASSERT(std::is_sorted(sorted.begin(), sorted.end()));

    // Invariant 2: Permutation preservation
    RC_ASSERT(std::is_permutation(sorted.begin(), sorted.end(), original.begin(), original.end()));
}
