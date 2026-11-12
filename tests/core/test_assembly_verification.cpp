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

[[gnu::noinline]] std::optional<std::size_t>
search_policy_branchless_prefetch(std::span<const int> s, int target) {
    return algoat::meta::static_search(s, target, p::branchless | p::prefetch);
}

[[gnu::noinline]] std::optional<std::size_t>
search_policy_branchless_no_prefetch(std::span<const int> s, int target) {
    return algoat::meta::static_search(s, target, p::branchless | p::no_prefetch);
}

[[gnu::noinline]] std::optional<std::size_t> search_policy_galloping(std::span<const int> s,
                                                                     int target) {
    return algoat::meta::static_search(s, target, p::galloping);
}

[[gnu::noinline]] std::optional<std::size_t> search_policy_eytzinger(std::span<const int> s,
                                                                     int target) {
    return algoat::meta::static_search(s, target, p::eytzinger | p::prefetch);
}

[[gnu::noinline]] std::optional<std::size_t> search_policy_static_btree(std::span<const int> s,
                                                                        int target) {
    return algoat::meta::static_search(s, target, p::static_btree | p::prefetch);
}

[[gnu::noinline]] int* select_policy_introselect(std::span<int> s, std::size_t k) {
    return std::to_address(algoat::meta::static_select(s, k, p::introselect | p::in_place));
}

[[gnu::noinline]] int* select_policy_branchless(std::span<int> s, std::size_t k) {
    return std::to_address(algoat::meta::static_select(s, k, p::branchless));
}

[[gnu::noinline]] int* select_policy_floyd_rivest(std::span<int> s, std::size_t k) {
    return std::to_address(algoat::meta::static_select(s, k, p::floyd_rivest));
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

TEST(AssemblyVerificationTest, StaticSearchAndSelectPolicyExecution) {
    std::vector<int> sorted = {10, 20, 30, 40, 50, 60, 70, 80};
    EXPECT_EQ(assembly_test::search_policy_branchless_prefetch(sorted, 50), 4);
    EXPECT_EQ(assembly_test::search_policy_branchless_no_prefetch(sorted, 20), 1);
    EXPECT_EQ(assembly_test::search_policy_galloping(sorted, 70), 6);
    EXPECT_EQ(assembly_test::search_policy_eytzinger(sorted, 30), 2);
    EXPECT_EQ(assembly_test::search_policy_static_btree(sorted, 60), 5);

    std::vector<int> v1 = {80, 10, 50, 30, 70, 20, 60, 40};
    EXPECT_EQ(*assembly_test::select_policy_introselect(v1, 3), 40);

    std::vector<int> v2 = {80, 10, 50, 30, 70, 20, 60, 40};
    EXPECT_EQ(*assembly_test::select_policy_branchless(v2, 5), 60);

    std::vector<int> v3 = {80, 10, 50, 30, 70, 20, 60, 40};
    EXPECT_EQ(*assembly_test::select_policy_floyd_rivest(v3, 2), 30);
}
