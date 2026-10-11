/**
 * @file test_policy_chain.cpp
 * @brief Unit tests for C++20 Policy Tag Chains, NTTP selectors, and static_sort.
 */

#include "algoat/algoat.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace p = algoat::meta::policy;

namespace {

// Compile-time constexpr evaluation test for static_sort
constexpr bool test_constexpr_static_sort() {
    std::array<int, 5> arr = {5, 3, 1, 4, 2};
    algoat::meta::static_sort(std::span<int, 5>{arr});
    for (std::size_t i = 0; i < 5; ++i) {
        if (arr[i] != static_cast<int>(i + 1)) {
            return false;
        }
    }

    std::array<int, 4> arr2 = {40, 10, 30, 20};
    algoat::meta::static_sort<p::stable | p::in_place>(std::span<int, 4>{arr2});
    return (arr2[0] == 10 && arr2[1] == 20 && arr2[2] == 30 && arr2[3] == 40);
}

static_assert(test_constexpr_static_sort(),
              "algoat::meta::static_sort must sort correctly in constexpr context.");

struct StablePayload {
    int key = 0;
    int original_index = 0;

    constexpr bool operator<(const StablePayload& other) const noexcept {
        return key < other.key;
    }
};

} // namespace

TEST(PolicyChainTest, TagCompositionAndDescriptorConversion) {
    // 1. Single tags
    constexpr auto d_stable = p::stable.operator algoat::meta::PolicyDescriptor();
    EXPECT_TRUE(d_stable.stable);
    EXPECT_FALSE(d_stable.in_place);

    constexpr auto d_inplace = p::in_place.operator algoat::meta::PolicyDescriptor();
    EXPECT_TRUE(d_inplace.in_place);
    EXPECT_FALSE(d_inplace.stable);

    // 2. Chaining two tags
    constexpr auto chain1 = p::stable | p::in_place;
    constexpr auto desc1 = chain1.to_descriptor();
    EXPECT_TRUE(desc1.stable);
    EXPECT_TRUE(desc1.in_place);
    EXPECT_FALSE(desc1.nearly_sorted);
    EXPECT_FALSE(desc1.force_radix);

    // 3. Commutativity of chaining
    constexpr auto chain2 = p::in_place | p::stable;
    constexpr auto desc2 = chain2.to_descriptor();
    EXPECT_EQ(desc1, desc2);

    // 4. Force radix tag combined with stable
    constexpr auto chain_radix = p::force_radix | p::stable;
    constexpr auto desc_radix = chain_radix.to_descriptor();
    EXPECT_TRUE(desc_radix.force_radix);
    EXPECT_TRUE(desc_radix.stable);
}

TEST(PolicyChainTest, ExampleUsageOneFromIssueSpecification) {
    // Exact syntax from Issue #32:
    // algoat::meta::static_sort(std::span{data}, p::stable | p::in_place);
    std::vector<int> data = {5, 2, 8, 1, 9};
    algoat::meta::static_sort(std::span{data}, p::stable | p::in_place);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data, (std::vector<int>{1, 2, 5, 8, 9}));
}

TEST(PolicyChainTest, ExampleUsageTwoFromIssueSpecification) {
    // Exact syntax from Issue #32:
    // std::array<int, 16> small_buf;
    // algoat::meta::static_sort(std::span<int, 16>{small_buf});
    std::array<int, 16> small_buf = {15, 3, 9, 1, 12, 6, 14, 2, 8, 4, 11, 7, 13, 5, 10, 0};
    algoat::meta::static_sort(std::span<int, 16>{small_buf});
    EXPECT_TRUE(std::is_sorted(small_buf.begin(), small_buf.end()));
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(small_buf[static_cast<std::size_t>(i)], i);
    }
}

TEST(PolicyChainTest, BlueprintTypeTemplateParameterAndDefault) {
    // Exact template signature from Issue #32:
    // template <typename Policy = PolicyChain<>, typename T, std::size_t N = std::dynamic_extent>
    // constexpr void static_sort(std::span<T, N> data, Policy = {});
    std::vector<int> data = {7, 1, 4, 2, 9};
    algoat::meta::static_sort<algoat::meta::PolicyChain<p::stable_t, p::in_place_t>>(
        std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    std::vector<int> data_default = {8, 3, 5, 1};
    algoat::meta::static_sort<>(std::span{data_default});
    EXPECT_TRUE(std::is_sorted(data_default.begin(), data_default.end()));
}

TEST(PolicyChainTest, StructuralClassNTTPInvocation) {
    // Direct structural class NTTP specification: static_sort<PolicyDescriptor{...}>(span)
    std::vector<int> data = {9, 1, 8, 2, 7, 3, 6, 4, 5};
    algoat::meta::static_sort<algoat::meta::PolicyDescriptor{.stable = true, .in_place = true}>(
        std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    // Tag chain used directly in NTTP parameter
    std::vector<int> data2 = {42, 17, 88, 3, 99, 1};
    algoat::meta::static_sort<p::stable | p::in_place>(std::span{data2});
    EXPECT_TRUE(std::is_sorted(data2.begin(), data2.end()));
}

TEST(PolicyChainTest, ForceRadixSortKernelOnIntegers) {
    // 1. Radix sort
    std::vector<int> data = {-50, 100, -2, 0, 42, -999, 500, 3};
    algoat::meta::static_sort(std::span{data}, p::force_radix);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    // 2. Stable Radix sort
    std::vector<int> data_stable = {99, -1, 45, 0, -88, 1234, 12};
    algoat::meta::static_sort(std::span{data_stable}, p::force_radix | p::stable);
    EXPECT_TRUE(std::is_sorted(data_stable.begin(), data_stable.end()));
}

TEST(PolicyChainTest, NearlySortedDataRoutesToAdaptiveSort) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 1);
    // Perturb a few elements
    std::swap(data[10], data[11]);
    std::swap(data[50], data[52]);

    algoat::meta::static_sort(std::span{data}, p::nearly_sorted);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(PolicyChainTest, StableSortPreservesRelativeOrder) {
    std::vector<StablePayload> data = {
        {3, 0}, {1, 1}, {3, 2}, {2, 3}, {1, 4}, {3, 5}, {2, 6}, {1, 7},
    };

    algoat::meta::static_sort(std::span{data}, p::stable | p::in_place);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    // Verify stability for key == 1
    std::vector<int> idx_for_1;
    for (const auto& item : data) {
        if (item.key == 1) {
            idx_for_1.push_back(item.original_index);
        }
    }
    EXPECT_EQ(idx_for_1, (std::vector<int>{1, 4, 7}));

    // Verify stability for key == 3
    std::vector<int> idx_for_3;
    for (const auto& item : data) {
        if (item.key == 3) {
            idx_for_3.push_back(item.original_index);
        }
    }
    EXPECT_EQ(idx_for_3, (std::vector<int>{0, 2, 5}));
}
