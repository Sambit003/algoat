#include "algoat/sorting/smoothsort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace algoat::sorting::testing {
namespace {

constexpr bool test_constexpr_smoothsort() {
    std::array<int, 10> arr = {9, 1, 5, 3, 7, 2, 8, 4, 6, 0};
    smoothsort(arr.begin(), arr.end());
    for (std::size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

static_assert(test_constexpr_smoothsort(), "smoothsort must be constexpr evaluatable");
static_assert(SortAlgorithm<SmoothSort, int>, "SmoothSort must satisfy SortAlgorithm concept");

TEST(SmoothSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    smoothsort(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());

    smoothsort(std::span<int>{empty});
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    smoothsort(single.begin(), single.end());
    ASSERT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);

    smoothsort(std::span<int>{single});
    EXPECT_EQ(single[0], 42);
}

TEST(SmoothSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    smoothsort(std::span<int>{sorted_pair});
    EXPECT_EQ(sorted_pair, (std::vector<int>{10, 20}));

    std::vector<int> reversed_pair = {20, 10};
    smoothsort(std::span<int>{reversed_pair});
    EXPECT_EQ(reversed_pair, (std::vector<int>{10, 20}));

    std::vector<int> equal_pair = {15, 15};
    smoothsort(std::span<int>{equal_pair});
    EXPECT_EQ(equal_pair, (std::vector<int>{15, 15}));
}

TEST(SmoothSortTest, ThreeElements) {
    std::vector<int> p = {1, 2, 3};
    do {
        auto copy = p;
        smoothsort(std::span<int>{copy});
        EXPECT_TRUE(std::is_sorted(copy.begin(), copy.end()));
    } while (std::next_permutation(p.begin(), p.end()));
}

TEST(SmoothSortTest, SmallArrayBelowLeonardoSizes) {
    for (std::size_t n = 4; n <= 32; ++n) {
        std::vector<int> data(n);
        std::iota(data.begin(), data.end(), 1);
        std::mt19937 rng(1337 + n);
        std::shuffle(data.begin(), data.end(), rng);

        smoothsort(std::span<int>{data});
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end())) << "Failed on size " << n;
    }
}

TEST(SmoothSortTest, PresortedLinearComparisons) {
    // Smoothsort should achieve strictly linear O(N) comparisons on presorted arrays
    for (std::size_t n : {64, 256, 1024, 4096}) {
        std::vector<int> data(n);
        std::iota(data.begin(), data.end(), 0);

        std::size_t comparisons = 0;
        auto counting_comp = [&comparisons](int a, int b) {
            ++comparisons;
            return a < b;
        };

        smoothsort(data.begin(), data.end(), counting_comp);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
        // On presorted inputs, Dijkstra proved smoothsort performs ~ 2N comparisons max
        EXPECT_LE(comparisons, 2 * n) << "Exceeded linear comparison bound on size " << n;
    }
}

TEST(SmoothSortTest, ReverseInputComparisonBound) {
    // Acceptance criteria: <= 1.5 * N * log2(N) comparisons on reverse inputs
    for (std::size_t n : {64, 256, 1024, 4096}) {
        std::vector<int> data(n);
        std::iota(data.rbegin(), data.rend(), 0);

        std::size_t comparisons = 0;
        auto counting_comp = [&comparisons](int a, int b) {
            ++comparisons;
            return a < b;
        };

        smoothsort(data.begin(), data.end(), counting_comp);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

        double max_allowed = 1.5 * static_cast<double>(n) * std::log2(static_cast<double>(n));
        EXPECT_LE(comparisons, static_cast<std::size_t>(max_allowed))
            << "Exceeded reverse comparison bound on size " << n << " (" << comparisons << " > "
            << max_allowed << ")";
    }
}

TEST(SmoothSortTest, DuplicatesAndIdenticalElements) {
    std::vector<int> all_same(500, 7);
    smoothsort(std::span<int>{all_same});
    EXPECT_TRUE(std::is_sorted(all_same.begin(), all_same.end()));

    std::vector<int> few_unique(1000);
    for (std::size_t i = 0; i < few_unique.size(); ++i) {
        few_unique[i] = static_cast<int>(i % 5);
    }
    smoothsort(std::span<int>{few_unique});
    EXPECT_TRUE(std::is_sorted(few_unique.begin(), few_unique.end()));
}

TEST(SmoothSortTest, LargeRandomArray) {
    constexpr std::size_t n = 20000;
    std::vector<int> data(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);
    for (auto& x : data) {
        x = dist(rng);
    }
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    smoothsort(data.begin(), data.end());
    EXPECT_EQ(data, expected);
}

TEST(SmoothSortTest, CustomComparatorAndProjection) {
    struct Item {
        int id;
        double score;
    };

    std::vector<Item> items = {{1, 3.5}, {2, 1.2}, {3, 9.8}, {4, 4.4}, {5, 0.5}, {6, 7.1}};

    // Sort descending by score using projection
    smoothsort(items, std::greater<double>{}, &Item::score);

    for (std::size_t i = 1; i < items.size(); ++i) {
        EXPECT_GE(items[i - 1].score, items[i].score);
    }
}

TEST(SmoothSortTest, AdapterStruct) {
    std::vector<int> data = {7, 2, 9, 1, 5, 4, 8, 3, 6};
    SmoothSort adapter;
    EXPECT_EQ(adapter.name(), "smoothsort");
    EXPECT_EQ(adapter.preferred_min_size(), 0u);

    adapter.sort(std::span<int>{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    adapter.sort(std::span<int>{data}, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
}

} // namespace
} // namespace algoat::sorting::testing
