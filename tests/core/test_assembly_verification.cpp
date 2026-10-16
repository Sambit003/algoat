/**
 * @file test_assembly_verification.cpp
 * @brief Verification of zero-overhead dispatch, compiler inlining, and elimination of vtables/jump
 * tables.
 */

#include "algoat/algoat.hpp"

#include <array>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

namespace p = algoat::meta::policy;

// Exported standalone functions compiled for assembly verification
namespace assembly_test {

[[gnu::noinline]] void sort_fixed_4(std::span<int, 4> s) {
    algoat::meta::static_sort(s);
}

[[gnu::noinline]] void sort_fixed_16(std::span<int, 16> s) {
    algoat::meta::static_sort(s);
}

[[gnu::noinline]] void sort_policy_chain(std::span<int> s) {
    algoat::meta::static_sort(s, p::stable | p::in_place);
}

[[gnu::noinline]] const int* search_branchless(const int* first, const int* last, int target) {
    return algoat::searching::branchless_lower_bound(first, last, target);
}

} // namespace assembly_test

TEST(AssemblyVerificationTest, InlinesFixedSizeSortingNetworkWithoutJumpTables) {
    std::array<int, 4> data = {4, 2, 3, 1};
    assembly_test::sort_fixed_4(std::span<int, 4>{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    std::array<int, 16> data16 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    assembly_test::sort_fixed_16(std::span<int, 16>{data16});
    EXPECT_TRUE(std::is_sorted(data16.begin(), data16.end()));
}

TEST(AssemblyVerificationTest, DirectStaticDispatchWithoutDynamicRegistry) {
    std::vector<int> data = {5, 3, 8, 1, 2};
    assembly_test::sort_policy_chain(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(AssemblyVerificationTest, BranchlessBinarySearchExecution) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60};
    const int* res = assembly_test::search_branchless(data.data(), data.data() + data.size(), 35);
    EXPECT_EQ(res, data.data() + 3);
    EXPECT_EQ(*res, 40);
}
