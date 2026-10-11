/**
 * @file test_radix_non_integral.cpp
 * @brief Negative compilation test: force_radix on non-integral types must fail static_assert.
 */

#include "algoat/algoat.hpp"

#include <span>
#include <string>
#include <vector>

int main() {
    namespace p = algoat::meta::policy;
    std::vector<std::string> words = {"banana", "apple", "cherry"};

    // Expected to fail static_assert:
    // "algoat::meta::static_sort: force_radix policy is only supported for integral types."
    algoat::meta::static_sort(std::span{words}, p::force_radix);

    return 0;
}
