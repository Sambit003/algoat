#include "algoat/sorting/pdqsort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace algoat::sorting::testing {
namespace {

struct Item {
    int id;
    int priority;

    constexpr auto operator<=>(const Item&) const = default;
};

// Compile-time constexpr test
constexpr bool test_constexpr_pdqsort() {
    std::array<int, 10> arr = {9, 1, 5, 3, 7, 2, 8, 4, 6, 0};
    pdqsort(arr.begin(), arr.end());
    for (std::size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

static_assert(test_constexpr_pdqsort(), "pdqsort must be constexpr evaluatable");

TEST(PdqSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    pdqsort(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());

    pdqsort(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    pdqsort(single.begin(), single.end());
    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);

    pdqsort(single);
    EXPECT_EQ(single[0], 42);
}

TEST(PdqSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    pdqsort(sorted_pair);
    EXPECT_EQ(sorted_pair, (std::vector<int>{10, 20}));

    std::vector<int> reversed_pair = {20, 10};
    pdqsort(reversed_pair);
    EXPECT_EQ(reversed_pair, (std::vector<int>{10, 20}));
}

TEST(PdqSortTest, SmallArrayInsertionSortBoundary) {
    // Tests sizes 3 through 32 (covering insertion_sort_threshold = 24)
    for (int size = 3; size <= 32; ++size) {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 1);
        std::reverse(data.begin(), data.end());

        pdqsort(data);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end())) << "Failed for size " << size;
    }
}

TEST(PdqSortTest, MediumArrayMedianOfThree) {
    // Sizes between 24 and 128 use median-of-3
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (int size : {30, 64, 100, 128}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        pdqsort(data);
        EXPECT_EQ(data, expected);
    }
}

TEST(PdqSortTest, BlockPartitioningBoundary) {
    // Sizes crossing block_size (64) and 2 * block_size (128)
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 100000);

    for (int size : {65, 127, 128, 129, 256, 512, 1024}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        pdqsort(data);
        EXPECT_EQ(data, expected);
    }
}

TEST(PdqSortTest, StrictlySortedLinearPath) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    pdqsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(PdqSortTest, StrictlyReverseSortedLinearPath) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end());

    pdqsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(PdqSortTest, PresortedNinetyFivePercent) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    std::mt19937 gen(99);
    std::uniform_int_distribution<std::size_t> idx_dist(0, N - 1);
    for (std::size_t i = 0; i < N / 20; ++i) { // swap 5% of elements
        std::swap(data[idx_dist(gen)], data[idx_dist(gen)]);
    }

    pdqsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(PdqSortTest, SingleValueDuplicates) {
    // Array where all elements are identical
    std::vector<int> data(10000, 7);
    pdqsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data.front(), 7);
    EXPECT_EQ(data.back(), 7);
}

TEST(PdqSortTest, ManyDuplicatesFewUniqueKeys) {
    // 3-way partition test
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    std::mt19937 gen(777);
    std::uniform_int_distribution<int> dist(0, 4); // Only 5 unique values

    for (int& x : data) {
        x = dist(gen);
    }

    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end());

    pdqsort(data);
    EXPECT_EQ(data, expected);
}

TEST(PdqSortTest, PipeOrganDistribution) {
    // 0, 1, 2, ..., N/2, ..., 2, 1, 0
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N / 2; ++i) {
        data[i] = static_cast<int>(i);
        data[N - 1 - i] = static_cast<int>(i);
    }

    pdqsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(PdqSortTest, SawtoothDistribution) {
    constexpr std::size_t N = 10000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i % 50);
    }

    pdqsort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(PdqSortTest, CustomComparatorDescending) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    pdqsort(data, std::greater<int>{});

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
    EXPECT_EQ(data.front(), 9);
    EXPECT_EQ(data.back(), 1);
}

TEST(PdqSortTest, CustomProjection) {
    std::vector<Item> items = {{1, 50}, {2, 10}, {3, 40}, {4, 20}, {5, 30}};

    pdqsort(items, std::ranges::less{}, &Item::priority);

    EXPECT_TRUE(std::is_sorted(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.priority < b.priority;
    }));
    EXPECT_EQ(items.front().priority, 10);
    EXPECT_EQ(items.back().priority, 50);
}

TEST(PdqSortTest, NonArithmeticTypesString) {
    std::vector<std::string> words = {"pattern",   "defeating", "quicksort", "branchless",
                                      "algorithm", "introsort", "heapsort",  "timsort",
                                      "blocksort", "radixsort"};

    pdqsort(words);
    EXPECT_TRUE(std::is_sorted(words.begin(), words.end()));
}

TEST(PdqSortTest, RandomArraysFrom100ToOneMillion) {
    // Tests sizes from 10^2 to 10^6 as specified in Acceptance Criteria
    std::mt19937_64 gen(2026);
    std::uniform_int_distribution<int> dist(-1'000'000, 1'000'000);

    for (std::size_t size : {100UL, 1'000UL, 10'000UL, 100'000UL, 1'000'000UL}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        pdqsort(data);
        EXPECT_EQ(data, expected) << "Failed sorting random array of size " << size;
    }
}

TEST(PdqSortTest, AdapterSortAlgorithmConcept) {
    static_assert(SortAlgorithm<PdqSort, int>);
    static_assert(SortAlgorithm<PDQSort, double>);

    PdqSort adapter;
    EXPECT_EQ(adapter.name(), "pdqsort");
    EXPECT_EQ(adapter.preferred_min_size(), 0u);

    std::vector<int> data = {9, 3, 7, 1, 5, 2, 8, 6, 4};
    adapter.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    adapter.sort(std::span{data}, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
}

} // namespace
} // namespace algoat::sorting::testing
