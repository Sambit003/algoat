/**
 * @file test_conflicting_stability_policies.cpp
 * @brief Negative compilation test: conflicting stability policies must fail static_assert.
 */

#include "algoat/algoat.hpp"

#include <span>
#include <vector>

int main() {
    namespace p = algoat::meta::policy;
    std::vector<int> data = {1, 2, 3};

    // Expected to fail static_assert:
    // "PolicyChain error: Conflicting stability policies specified (stable vs unstable)."
    algoat::meta::static_sort(std::span{data}, p::stable | p::unstable);

    return 0;
}
