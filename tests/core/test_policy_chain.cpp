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

TEST(PolicyChainTest, SearchAndSelectDescriptorComposition) {
    // Default structural aggregates
    constexpr algoat::meta::SearchPolicyDescriptor default_search{};
    static_assert(std::is_aggregate_v<algoat::meta::SearchPolicyDescriptor>);
    static_assert(!default_search.branchless);
    static_assert(default_search.prefetch);
    static_assert(!default_search.eytzinger);
    static_assert(!default_search.static_btree);
    static_assert(!default_search.galloping);
    EXPECT_EQ(default_search, algoat::meta::SearchPolicyDescriptor{});

    constexpr algoat::meta::SelectPolicyDescriptor default_select{};
    static_assert(std::is_aggregate_v<algoat::meta::SelectPolicyDescriptor>);
    static_assert(default_select.in_place);
    static_assert(default_select.guaranteed_linear);
    static_assert(!default_select.floyd_rivest);
    static_assert(!default_select.branchless);
    EXPECT_EQ(default_select, algoat::meta::SelectPolicyDescriptor{});

    // Search policy tag chaining
    constexpr auto search_chain = p::branchless | p::prefetch;
    constexpr auto search_desc = search_chain.to_search_descriptor();
    static_assert(search_desc.branchless);
    static_assert(search_desc.prefetch);
    static_assert(!search_desc.eytzinger);
    EXPECT_TRUE(search_desc.branchless);
    EXPECT_TRUE(search_desc.prefetch);

    constexpr auto eytzinger_no_prefetch = (p::eytzinger | p::no_prefetch).to_search_descriptor();
    static_assert(eytzinger_no_prefetch.eytzinger);
    static_assert(!eytzinger_no_prefetch.prefetch);

    // Select policy tag chaining
    constexpr auto select_chain = p::introselect | p::in_place;
    constexpr auto select_desc = select_chain.to_select_descriptor();
    static_assert(select_desc.in_place);
    static_assert(select_desc.guaranteed_linear);
    static_assert(!select_desc.floyd_rivest);
    static_assert(!select_desc.branchless);
    EXPECT_TRUE(select_desc.in_place);
    EXPECT_TRUE(select_desc.guaranteed_linear);

    constexpr auto select_branchless = (p::branchless | p::in_place).to_select_descriptor();
    static_assert(select_branchless.branchless);
    static_assert(select_branchless.in_place);
}

constexpr bool test_constexpr_static_search() {
    constexpr std::array<int, 7> arr = {2, 4, 6, 8, 10, 12, 14};
    if (algoat::meta::static_search(std::span<const int, 7>{arr}, 8) != 3) {
        return false;
    }
    if (algoat::meta::static_search(std::span<const int, 7>{arr}, 7).has_value()) {
        return false;
    }
    if (algoat::meta::static_search<p::branchless | p::no_prefetch>(std::span<const int, 7>{arr},
                                                                    12) != 5) {
        return false;
    }
    if (algoat::meta::static_search(std::span<const int, 7>{arr}, 2, p::galloping) != 0) {
        return false;
    }
    if (algoat::meta::static_search(std::span<const int, 7>{arr}, 10, p::eytzinger) != 4) {
        return false;
    }
    if (algoat::meta::static_search(std::span<const int, 7>{arr}, 14, p::static_btree) != 6) {
        return false;
    }
    return true;
}

static_assert(test_constexpr_static_search(),
              "algoat::meta::static_search must evaluate correctly in constexpr context.");

TEST(PolicyChainTest, StaticSearchPolicyDispatch) {
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    std::span<const int> span{data};

    // Default policy
    EXPECT_EQ(algoat::meta::static_search(span, 7), 3);
    EXPECT_EQ(algoat::meta::static_search(span, 8), std::nullopt);

    // Direct structural NTTP
    EXPECT_EQ(
        (algoat::meta::static_search<algoat::meta::SearchPolicyDescriptor{.branchless = true}>(span,
                                                                                               11)),
        5);

    // Tag chain NTTP & runtime argument syntax
    EXPECT_EQ((algoat::meta::static_search<p::branchless | p::prefetch>(span, 15)), 7);
    EXPECT_EQ(algoat::meta::static_search(span, 15, p::branchless | p::prefetch), 7);
    EXPECT_EQ(algoat::meta::static_search(span, 5, p::branchless | p::no_prefetch), 2);
    EXPECT_EQ(algoat::meta::static_search(span, 2, p::branchless | p::prefetch), std::nullopt);

    // Single-tag overloads & layout kernels
    EXPECT_EQ(algoat::meta::static_search(span, 13, p::eytzinger), 6);
    EXPECT_EQ(algoat::meta::static_search(span, 14, p::eytzinger), std::nullopt);
    EXPECT_EQ(algoat::meta::static_search(span, 19, p::eytzinger | p::no_prefetch), 9);

    EXPECT_EQ(algoat::meta::static_search(span, 9, p::static_btree), 4);
    EXPECT_EQ(algoat::meta::static_search(span, 10, p::static_btree), std::nullopt);
    EXPECT_EQ(algoat::meta::static_search(span, 1, p::static_btree | p::prefetch), 0);

    EXPECT_EQ(algoat::meta::static_search(span, 3, p::galloping), 1);
    EXPECT_EQ(algoat::meta::static_search(span, 4, p::galloping), std::nullopt);

    // Empty range
    std::vector<int> empty;
    EXPECT_EQ(algoat::meta::static_search(empty, 42, p::branchless | p::prefetch), std::nullopt);
    EXPECT_EQ(algoat::meta::static_search(empty, 42, p::eytzinger), std::nullopt);
    EXPECT_EQ(algoat::meta::static_search(empty, 42, p::static_btree), std::nullopt);

    // Heterogeneous keys must never suffer lossy narrowing
    std::vector<int> ints{1, 3, 5};
    EXPECT_EQ(algoat::meta::static_search(ints, 3.5, p::eytzinger), std::nullopt);
    EXPECT_EQ(algoat::meta::static_search(ints, 3.5, p::static_btree), std::nullopt);
    const long long wide_key = 1LL << 33;
    EXPECT_EQ(algoat::meta::static_search(ints, wide_key, p::eytzinger), std::nullopt);
    EXPECT_EQ(algoat::meta::static_search(ints, wide_key, p::static_btree), std::nullopt);
}

constexpr bool test_constexpr_static_select() {
    std::array<int, 7> a{42, 7, 19, 3, 25, 11, 19};
    auto a1 = a;
    auto it1 = algoat::meta::static_select(a1, 3, p::introselect | p::in_place);
    if (it1 != a1.begin() + 3 || *it1 != 19) {
        return false;
    }

    auto a2 = a;
    auto it2 = algoat::meta::static_select(a2, 0, p::branchless);
    if (it2 != a2.begin() || *it2 != 3) {
        return false;
    }

    auto a3 = a;
    auto it3 = algoat::meta::static_select<algoat::meta::SelectPolicyDescriptor{
        .guaranteed_linear = false, .floyd_rivest = true}>(a3, 6);
    if (it3 != a3.begin() + 6 || *it3 != 42) {
        return false;
    }

    std::array<int, 2> small{5, 1};
    auto it_oob = algoat::meta::static_select(small, 5, p::introselect);
    if (it_oob != small.end()) {
        return false;
    }

    return true;
}

static_assert(test_constexpr_static_select());

TEST(PolicyChainTest, StaticSelectPolicyDispatch) {
    const std::vector<int> base{9, 1, 5, 3, 7, 5, 2, 8, 6, 4, 5, 0, 12, 11, 10};
    auto verify_select = [&](auto policy_fn) {
        for (std::size_t k = 0; k < base.size(); ++k) {
            std::vector<int> data = base;
            auto it = policy_fn(data, k);
            ASSERT_EQ(it, data.begin() + static_cast<std::ptrdiff_t>(k));

            std::vector<int> expected = base;
            std::ranges::sort(expected);
            EXPECT_EQ(*it, expected[k]);

            for (auto p_it = data.begin(); p_it < it; ++p_it) {
                EXPECT_LE(*p_it, *it);
            }
            for (auto p_it = it + 1; p_it < data.end(); ++p_it) {
                EXPECT_GE(*p_it, *it);
            }

            std::ranges::sort(data);
            EXPECT_EQ(data, expected);
        }
    };

    // Default dispatch
    verify_select(
        [](std::vector<int>& v, std::size_t k) { return algoat::meta::static_select(v, k); });

    // Declarative policy chains and tags
    verify_select([](std::vector<int>& v, std::size_t k) {
        return algoat::meta::static_select(v, k, p::introselect | p::in_place);
    });
    verify_select([](std::vector<int>& v, std::size_t k) {
        return algoat::meta::static_select(v, k, p::branchless);
    });
    verify_select([](std::vector<int>& v, std::size_t k) {
        return algoat::meta::static_select(v, k, p::floyd_rivest);
    });
    verify_select([](std::vector<int>& v, std::size_t k) {
        return algoat::meta::static_select(v, k, p::floyd_rivest | p::in_place);
    });

    // Direct structural NTTP dispatch (including guaranteed_linear = false)
    verify_select([](std::vector<int>& v, std::size_t k) {
        return algoat::meta::static_select<algoat::meta::SelectPolicyDescriptor{.branchless =
                                                                                    true}>(v, k);
    });
    verify_select([](std::vector<int>& v, std::size_t k) {
        return algoat::meta::static_select<algoat::meta::SelectPolicyDescriptor{
            .guaranteed_linear = false, .floyd_rivest = true}>(v, k);
    });

    // Out-of-bounds and empty range handling
    std::vector<int> empty;
    EXPECT_EQ(algoat::meta::static_select(empty, 0, p::introselect), empty.end());
    EXPECT_EQ(algoat::meta::static_select(empty, 0, p::floyd_rivest), empty.end());
    EXPECT_EQ(algoat::meta::static_select(empty, 0, p::branchless), empty.end());

    std::vector<int> non_empty{3, 1, 2};
    EXPECT_EQ(algoat::meta::static_select(non_empty, 3, p::floyd_rivest), non_empty.end());
    EXPECT_EQ(algoat::meta::static_select(non_empty, 10, p::branchless), non_empty.end());
}

TEST(PolicyChainTest, LargeRangeAndNonContiguousDispatch) {
    // Exercise multi-level StaticBTree / Eytzinger prefetching and Floyd-Rivest >600 path.
    constexpr std::size_t N = 1500;
    std::vector<int> sorted(N);
    for (std::size_t i = 0; i < N; ++i) {
        sorted[i] = static_cast<int>(i * 3);
    }

    for (std::size_t i = 0; i < N; i += 97) {
        int hit = static_cast<int>(i * 3);
        int miss = hit + 1;
        EXPECT_EQ(algoat::meta::static_search(sorted, hit, p::branchless | p::prefetch), i);
        EXPECT_EQ(algoat::meta::static_search(sorted, miss, p::branchless | p::prefetch),
                  std::nullopt);
        EXPECT_EQ(algoat::meta::static_search(sorted, hit, p::eytzinger | p::prefetch), i);
        EXPECT_EQ(algoat::meta::static_search(sorted, miss, p::eytzinger | p::no_prefetch),
                  std::nullopt);
        EXPECT_EQ(algoat::meta::static_search(sorted, hit, p::static_btree | p::prefetch), i);
        EXPECT_EQ(algoat::meta::static_search(sorted, miss, p::static_btree | p::no_prefetch),
                  std::nullopt);
        EXPECT_EQ(algoat::meta::static_search(sorted, hit, p::galloping), i);
    }

    // Large >600 selection across duplicate-heavy and organ-pipe patterns
    std::vector<int> pipe(N);
    for (std::size_t i = 0; i < N / 2; ++i) {
        pipe[i] = static_cast<int>(i % 19);
        pipe[N - 1 - i] = static_cast<int>(i % 19);
    }
    std::vector<int> expected = pipe;
    std::ranges::sort(expected);

    for (std::size_t k : {std::size_t{0}, std::size_t{17}, N / 2, N - 1}) {
        auto v1 = pipe;
        auto it1 = algoat::meta::static_select(v1, k, p::floyd_rivest);
        EXPECT_EQ(*it1, expected[k]);
        auto v2 = pipe;
        auto it2 = algoat::meta::static_select(v2, k, p::branchless);
        EXPECT_EQ(*it2, expected[k]);
        auto v3 = pipe;
        auto it3 = algoat::meta::static_select<algoat::meta::SelectPolicyDescriptor{
            .guaranteed_linear = false, .floyd_rivest = true}>(v3, k);
        EXPECT_EQ(*it3, expected[k]);
    }
}
