#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::sorting;

// Compile-time concept verification
static_assert(SortAlgorithm<QuickSort, int>);
static_assert(SortAlgorithm<QuickSort, double>);

namespace {
struct MoveOnlyInt {
    int value;

    explicit MoveOnlyInt(int v) : value(v) {}
    MoveOnlyInt(const MoveOnlyInt&) = delete;
    MoveOnlyInt& operator=(const MoveOnlyInt&) = delete;
    MoveOnlyInt(MoveOnlyInt&&) noexcept = default;
    MoveOnlyInt& operator=(MoveOnlyInt&&) noexcept = default;

    auto operator<=>(const MoveOnlyInt& other) const = default;
};
} // namespace

class QuickSortTest : public ::testing::Test {
protected:
    QuickSort algo;

    template <typename T> void verify_sort(std::vector<T>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

TEST_F(QuickSortTest, TwoElements) {
    std::vector<int> sorted = {10, 20};
    verify_sort(sorted);

    std::vector<int> reversed = {20, 10};
    verify_sort(reversed);

    std::vector<int> duplicate = {15, 15};
    verify_sort(duplicate);
}

TEST_F(QuickSortTest, ThreeElementsPermutations) {
    std::vector<std::vector<int>> perms = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2},
                                           {3, 2, 1}, {1, 1, 2}, {2, 1, 1}, {1, 2, 1}, {5, 5, 5}};
    for (auto& vec : perms) {
        verify_sort(vec);
    }
}

TEST_F(QuickSortTest, AllDuplicatesLarge) {
    // 100,000 identical elements — tests O(N) linear time on duplicates and no stack overflow
    std::vector<int> data(100000, 42);
    verify_sort(data);
}

TEST_F(QuickSortTest, FewUniqueValues) {
    // Low-entropy dataset with only 3 distinct values across 50,000 elements
    std::vector<int> data(50000);
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int> dist(0, 2);
    for (int& x : data) {
        x = dist(gen);
    }
    verify_sort(data);
}

TEST_F(QuickSortTest, OrganPipeDistribution) {
    // Ascending to midpoint, then descending
    const std::size_t n = 10000;
    std::vector<int> data(n);
    for (std::size_t i = 0; i < n / 2; ++i) {
        data[i] = static_cast<int>(i);
        data[n - 1 - i] = static_cast<int>(i);
    }
    verify_sort(data);
}

TEST_F(QuickSortTest, SawtoothDistribution) {
    // Repeating ascending ramps with heavy duplicate density
    const std::size_t n = 10000;
    std::vector<int> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<int>(i % 50);
    }
    verify_sort(data);
}

TEST_F(QuickSortTest, DitheredDuplicates) {
    // Alternating elements (0, 1, 0, 1, ...)
    const std::size_t n = 10000;
    std::vector<int> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<int>(i % 2);
    }
    verify_sort(data);
}

TEST_F(QuickSortTest, NegativeAndFloatingTypes) {
    std::vector<int> signed_data = {-500, 20, -10, 0, 4, -100, 25, -500, 30};
    verify_sort(signed_data);

    std::vector<double> float_data = {3.14, -2.71, 0.0, 1.414, -0.577, 3.14, -100.25};
    verify_sort(float_data);
}

TEST_F(QuickSortTest, MoveOnlyTypes) {
    std::vector<MoveOnlyInt> data;
    data.emplace_back(50);
    data.emplace_back(10);
    data.emplace_back(30);
    data.emplace_back(20);
    data.emplace_back(40);
    data.emplace_back(10);
    data.emplace_back(30);

    algo.sort(std::span{data});

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data[0].value, 10);
    EXPECT_EQ(data[1].value, 10);
    EXPECT_EQ(data[2].value, 20);
    EXPECT_EQ(data[3].value, 30);
    EXPECT_EQ(data[4].value, 30);
    EXPECT_EQ(data[5].value, 40);
    EXPECT_EQ(data[6].value, 50);
}

TEST_F(QuickSortTest, LargeRandom) {
    std::vector<int> data(100000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (int& x : data) {
        x = dist(gen);
    }
    verify_sort(data);
}

TEST_F(QuickSortTest, PreferredMinSizeAndName) {
    EXPECT_EQ(algo.name(), "quicksort");
    EXPECT_EQ(algo.preferred_min_size(), 32);
}

TEST(QuickSortSimdIntegration, LargeInt32RandomArraySortsCorrectly) {
    std::vector<int32_t> data(500);
    std::iota(data.begin(), data.end(), 0);
    std::shuffle(data.begin(), data.end(), std::mt19937{42});

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(QuickSortSimdIntegration, LargeInt64RandomArraySortsCorrectly) {
    std::vector<int64_t> data(1000);
    std::iota(data.begin(), data.end(), 0LL);
    std::shuffle(data.begin(), data.end(), std::mt19937{99});

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(QuickSortSimdIntegration, LargeUInt32ReverseSortedSortsCorrectly) {
    std::vector<uint32_t> data(512);
    std::iota(data.begin(), data.end(), 0u);
    std::reverse(data.begin(), data.end());

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(QuickSortSimdIntegration, LargeAllDuplicatesFallsBackToScalar) {
    // All-duplicate array triggers the arr[low]==pivot guard; must not hang or corrupt.
    std::vector<int32_t> data(500, 42);

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data.front(), 42);
    EXPECT_EQ(data.back(), 42);
}

TEST(QuickSortSimdIntegration, LargeNegativeInt64ArraySortsCorrectly) {
    std::vector<int64_t> data(300);
    std::iota(data.begin(), data.end(), -150LL);
    std::shuffle(data.begin(), data.end(), std::mt19937{7});

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(QuickSortSimdIntegration, LargeAlreadySortedArraySortsCorrectly) {
    // Tests boundary condition where all elements in partition sub-ranges are already ordered
    std::vector<int32_t> data(1000);
    std::iota(data.begin(), data.end(), 0);

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(QuickSortSimdIntegration, LargeNearlySortedArraySortsCorrectly) {
    std::vector<int32_t> data(1000);
    std::iota(data.begin(), data.end(), 0);
    for (std::size_t i = 0; i + 10 < data.size(); i += 50) {
        std::swap(data[i], data[i + 10]);
    }

    QuickSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
