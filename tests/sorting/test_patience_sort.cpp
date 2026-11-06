#include "algoat/sorting/patience_sort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace algoat::sorting::testing {
namespace {

static_assert(SortAlgorithm<PatienceSort, int>, "PatienceSort must satisfy SortAlgorithm concept");

// Independent DP reference for strictly increasing LIS length
template <typename T, typename Comp = std::ranges::less>
std::size_t reference_lis_length(std::span<const T> data, Comp comp = {}) {
    if (data.empty()) {
        return 0;
    }
    std::vector<T> tails;
    for (const auto& x : data) {
        auto it = std::ranges::lower_bound(tails, x, comp);
        if (it == tails.end()) {
            tails.push_back(x);
        } else {
            *it = x;
        }
    }
    return tails.size();
}

TEST(PatienceSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    auto res_empty = patience_sort(empty);
    EXPECT_TRUE(res_empty.sorted_data.empty());
    EXPECT_TRUE(res_empty.lis_indices.empty());

    std::vector<int> single = {42};
    auto res_single = patience_sort(single);
    ASSERT_EQ(res_single.sorted_data.size(), 1u);
    EXPECT_EQ(res_single.sorted_data[0], 42);
    ASSERT_EQ(res_single.lis_indices.size(), 1u);
    EXPECT_EQ(res_single.lis_indices[0], 0u);
}

TEST(PatienceSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    auto res_sorted = patience_sort(sorted_pair);
    EXPECT_EQ(res_sorted.sorted_data, (std::vector<int>{10, 20}));
    EXPECT_EQ(res_sorted.lis_indices, (std::vector<std::size_t>{0, 1}));

    std::vector<int> reversed_pair = {20, 10};
    auto res_rev = patience_sort(reversed_pair);
    EXPECT_EQ(res_rev.sorted_data, (std::vector<int>{10, 20}));
    EXPECT_EQ(res_rev.lis_indices.size(), 1u);

    std::vector<int> equal_pair = {15, 15};
    auto res_eq = patience_sort(equal_pair);
    EXPECT_EQ(res_eq.sorted_data, (std::vector<int>{15, 15}));
    EXPECT_EQ(res_eq.lis_indices.size(), 1u);
}

TEST(PatienceSortTest, StrictlyAscendingArray) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    auto res = patience_sort(data);
    EXPECT_EQ(res.sorted_data, data);
    EXPECT_EQ(res.lis_indices.size(), data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(res.lis_indices[i], i);
    }
}

TEST(PatienceSortTest, StrictlyDescendingArray) {
    std::vector<int> data = {8, 7, 6, 5, 4, 3, 2, 1};
    auto res = patience_sort(data);
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(res.sorted_data, expected);
    EXPECT_EQ(res.lis_indices.size(), 1u);
}

TEST(PatienceSortTest, ClassicalAldousDiaconisExample) {
    // Classic example from Aldous & Diaconis (1999)
    std::vector<int> data = {7, 2, 8, 1, 3, 4, 10, 6, 9, 5};
    auto res = patience_sort(data);

    auto expected = data;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(res.sorted_data, expected);

    // Verify LIS validity
    ASSERT_EQ(res.lis_indices.size(), reference_lis_length<int>(data));
    for (std::size_t i = 1; i < res.lis_indices.size(); ++i) {
        EXPECT_LT(res.lis_indices[i - 1], res.lis_indices[i]);
        EXPECT_LT(data[res.lis_indices[i - 1]], data[res.lis_indices[i]]);
    }
}

TEST(PatienceSortTest, DuplicateElements) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    auto res = patience_sort(data);

    auto expected = data;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(res.sorted_data, expected);

    // LIS with duplicates must be strictly increasing
    ASSERT_EQ(res.lis_indices.size(), reference_lis_length<int>(data));
    for (std::size_t i = 1; i < res.lis_indices.size(); ++i) {
        EXPECT_LT(res.lis_indices[i - 1], res.lis_indices[i]);
        EXPECT_LT(data[res.lis_indices[i - 1]], data[res.lis_indices[i]]);
    }
}

TEST(PatienceSortTest, LargeRandomArrayWithLisCheck) {
    constexpr std::size_t n = 5000;
    std::vector<int> data(n);
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (auto& x : data) {
        x = dist(rng);
    }

    auto res = patience_sort(data);

    auto expected = data;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(res.sorted_data, expected);

    const std::size_t true_lis = reference_lis_length<int>(data);
    EXPECT_EQ(res.lis_indices.size(), true_lis);

    for (std::size_t i = 1; i < res.lis_indices.size(); ++i) {
        EXPECT_LT(res.lis_indices[i - 1], res.lis_indices[i]);
        EXPECT_LT(data[res.lis_indices[i - 1]], data[res.lis_indices[i]]);
    }
}

TEST(PatienceSortTest, CustomComparatorLongestDecreasingSubsequence) {
    std::vector<int> data = {3, 10, 2, 1, 20};
    // Using std::greater sorts descending and computes the Longest Decreasing Subsequence
    auto res = patience_sort(data, std::greater<int>{});

    std::vector<int> expected = {20, 10, 3, 2, 1};
    EXPECT_EQ(res.sorted_data, expected);

    for (std::size_t i = 1; i < res.lis_indices.size(); ++i) {
        EXPECT_LT(res.lis_indices[i - 1], res.lis_indices[i]);
        EXPECT_GT(data[res.lis_indices[i - 1]], data[res.lis_indices[i]]);
    }
}

TEST(PatienceSortTest, FiftyThousandPermutationsLisVerification) {
    // Acceptance criterion: LIS output verified against classic O(N log N) dynamic programming
    // across 50,000 random permutations.
    constexpr std::size_t kTotalPermutations = 50000;
    std::mt19937 rng(424242);
    std::vector<int> buffer;
    buffer.reserve(64);

    for (std::size_t trial = 0; trial < kTotalPermutations; ++trial) {
        const std::size_t n = 5 + (trial % 35); // lengths from 5 to 39
        buffer.resize(n);
        std::iota(buffer.begin(), buffer.end(), 1);
        std::shuffle(buffer.begin(), buffer.end(), rng);

        auto res = patience_sort(buffer);

        const std::size_t expected_lis = reference_lis_length<int>(buffer);
        ASSERT_EQ(res.lis_indices.size(), expected_lis)
            << "LIS length mismatch on trial " << trial << " with length " << n;

        for (std::size_t i = 1; i < res.lis_indices.size(); ++i) {
            EXPECT_LT(res.lis_indices[i - 1], res.lis_indices[i]);
            EXPECT_LT(buffer[res.lis_indices[i - 1]], buffer[res.lis_indices[i]]);
        }

        EXPECT_TRUE(std::is_sorted(res.sorted_data.begin(), res.sorted_data.end()));
    }
}

TEST(PatienceSortTest, AdapterStruct) {
    std::vector<int> data = {9, 3, 7, 1, 8, 2, 6, 4, 5};
    PatienceSort adapter;
    EXPECT_EQ(adapter.name(), "patiencesort");
    EXPECT_EQ(adapter.preferred_min_size(), 0u);

    adapter.sort(std::span<int>{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    adapter.sort(std::span<int>{data}, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
}

} // namespace
} // namespace algoat::sorting::testing
