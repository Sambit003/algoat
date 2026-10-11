/**
 * @file test_duplicate_policies.cpp
 * @brief Negative compilation test: duplicate policy tags must fail static_assert.
 */

#include "algoat/algoat.hpp"

#include <span>
#include <vector>

int main() {
    namespace p = algoat::meta::policy;
    std::vector<int> data = {1, 2, 3};

    // Expected to fail static_assert:
    // "PolicyChain error: Duplicate policy tags specified in chain."
    algoat::meta::static_sort(std::span{data}, p::stable | p::stable);

    return 0;
}
