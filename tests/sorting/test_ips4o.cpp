#include "algoat/sorting/ips4o.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
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

// Concept verification
static_assert(SortAlgorithm<Ips4oSort, int>, "Ips4oSort must satisfy SortAlgorithm concept");
static_assert(SortAlgorithm<Ips4oSort, double>, "Ips4oSort must satisfy SortAlgorithm concept");

TEST(Ips4oSortTest, ConceptAndTraits) {
    Ips4oSort sorter;
    EXPECT_EQ(sorter.name(), "ips4o");
    EXPECT_EQ(sorter.preferred_min_size(), 65536u);
}

TEST(Ips4oSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    ips4o_parallel_sort(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());

    ips4o_parallel_sort(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    ips4o_parallel_sort(single.begin(), single.end());
    ASSERT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);

    ips4o_parallel_sort(single);
    EXPECT_EQ(single[0], 42);
}

TEST(Ips4oSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    ips4o_parallel_sort(sorted_pair);
    EXPECT_EQ(sorted_pair, (std::vector<int>{10, 20}));

    std::vector<int> reversed_pair = {20, 10};
    ips4o_parallel_sort(reversed_pair);
    EXPECT_EQ(reversed_pair, (std::vector<int>{10, 20}));
}

TEST(Ips4oSortTest, SmallArrayBaseCase) {
    for (int size = 3; size <= 128; ++size) {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 1);
        std::reverse(data.begin(), data.end());

        ips4o_parallel_sort(data);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end())) << "Failed for size " << size;
    }
}

TEST(Ips4oSortTest, MediumArray) {
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(-10000, 10000);

    for (int size : {256, 512, 1024, 4096, 16384}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        ips4o_parallel_sort(data);
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(Ips4oSortTest, LargeArrayAtAndAboveThreshold) {
    // 65536 is the dispatcher parallel threshold
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-500000, 500000);

    for (int size : {65535, 65536, 100000}) {
        std::vector<int> data(size);
        for (int& x : data) {
            x = dist(gen);
        }
        std::vector<int> expected = data;
        std::sort(expected.begin(), expected.end());

        ips4o_parallel_sort(data);
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

TEST(Ips4oSortTest, PathologicalDistributions) {
    constexpr int size = 70000;
    std::vector<int> data(size);

    // 1. All duplicates (triggers equality buckets)
    std::fill(data.begin(), data.end(), 42);
    ips4o_parallel_sort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    // 2. Heavy duplicates (few distinct values)
    std::mt19937 gen(999);
    std::uniform_int_distribution<int> small_range(1, 8);
    for (int& x : data) {
        x = small_range(gen);
    }
    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end());
    ips4o_parallel_sort(data);
    EXPECT_EQ(data, expected);

    // 3. Already sorted
    std::iota(data.begin(), data.end(), 0);
    ips4o_parallel_sort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    // 4. Reverse sorted
    std::reverse(data.begin(), data.end());
    ips4o_parallel_sort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    // 5. Pipe-organ (asc then desc)
    for (int i = 0; i < size / 2; ++i) {
        data[i] = i;
        data[size - 1 - i] = i;
    }
    expected = data;
    std::sort(expected.begin(), expected.end());
    ips4o_parallel_sort(data);
    EXPECT_EQ(data, expected);

    // 6. Sawtooth
    for (int i = 0; i < size; ++i) {
        data[i] = i % 100;
    }
    expected = data;
    std::sort(expected.begin(), expected.end());
    ips4o_parallel_sort(data);
    EXPECT_EQ(data, expected);
}

TEST(Ips4oSortTest, CustomComparatorDescending) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    ips4o_parallel_sort(data, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));

    // Large array with custom comparator
    constexpr int size = 68000;
    data.resize(size);
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> dist(-100000, 100000);
    for (int& x : data) {
        x = dist(gen);
    }
    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end(), std::greater<int>{});
    ips4o_parallel_sort(data, std::greater<int>{});
    EXPECT_EQ(data, expected);
}

TEST(Ips4oSortTest, CustomProjection) {
    std::vector<Item> items = {{1, 50}, {2, 10}, {3, 40}, {4, 20}, {5, 30}};
    ips4o_parallel_sort(items, std::less<int>{}, &Item::priority);

    EXPECT_EQ(items[0].priority, 10);
    EXPECT_EQ(items[1].priority, 20);
    EXPECT_EQ(items[2].priority, 30);
    EXPECT_EQ(items[3].priority, 40);
    EXPECT_EQ(items[4].priority, 50);

    // Large projected array
    constexpr int size = 66000;
    items.resize(size);
    std::mt19937 gen(54321);
    std::uniform_int_distribution<int> dist(0, 1000000);
    for (int i = 0; i < size; ++i) {
        items[i] = Item{i, dist(gen)};
    }
    std::vector<Item> expected = items;
    std::sort(expected.begin(), expected.end(),
              [](const Item& a, const Item& b) { return a.priority < b.priority; });
    ips4o_parallel_sort(items, std::less<int>{}, &Item::priority);
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(items[i].priority, expected[i].priority);
    }
}

TEST(Ips4oSortTest, MoveOnlyTypes) {
    constexpr int size = 1000;
    std::vector<std::unique_ptr<int>> data;
    data.reserve(size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 10000);
    for (int i = 0; i < size; ++i) {
        data.push_back(std::make_unique<int>(dist(gen)));
    }

    auto comp = [](const std::unique_ptr<int>& a, const std::unique_ptr<int>& b) {
        return *a < *b;
    };

    ips4o_parallel_sort(data, comp);

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), comp));
}

TEST(Ips4oSortTest, AdapterSortSpan) {
    constexpr int size = 70000;
    std::vector<int> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end());

    Ips4oSort sorter;
    sorter.sort(std::span<int>(data.data(), data.size()));
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(Ips4oSortTest, LargeOneMillionElements) {
    constexpr int size = 1000000;
    std::vector<int> data(size);
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(0, 10000000);
    for (int& x : data) {
        x = dist(gen);
    }
    std::vector<int> expected = data;
    std::sort(expected.begin(), expected.end());

    ips4o_parallel_sort(data);

    EXPECT_EQ(data, expected);
}

} // namespace
} // namespace algoat::sorting::testing
