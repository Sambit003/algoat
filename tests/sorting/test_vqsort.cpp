#include "algoat/sorting/sorting.hpp"
#include "algoat/sorting/vqsort.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using algoat::sorting::SortAlgorithm;
using algoat::sorting::SortVariant;
using algoat::sorting::vqsort;
using algoat::sorting::VQSort;

// 1. Concept Compliance
static_assert(SortAlgorithm<VQSort, int>);
static_assert(SortAlgorithm<VQSort, uint32_t>);
static_assert(SortAlgorithm<VQSort, int64_t>);
static_assert(SortAlgorithm<VQSort, uint64_t>);
static_assert(SortAlgorithm<VQSort, float>);
static_assert(SortAlgorithm<VQSort, double>);

TEST(VQSortTest, ConceptAndName) {
    VQSort algo;
    EXPECT_EQ(algo.name(), "vqsort");
    EXPECT_EQ(algo.preferred_min_size(), 16u);
}

// 2. Empty and Single-Element Arrays
TEST(VQSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    vqsort(std::span{empty});
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    vqsort(std::span{single});
    ASSERT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);
}

// 3. Two Elements
TEST(VQSortTest, TwoElements) {
    std::vector<int> pair_desc = {99, 12};
    vqsort(std::span{pair_desc});
    EXPECT_EQ(pair_desc[0], 12);
    EXPECT_EQ(pair_desc[1], 99);

    std::vector<int> pair_asc = {12, 99};
    vqsort(std::span{pair_asc});
    EXPECT_EQ(pair_asc[0], 12);
    EXPECT_EQ(pair_asc[1], 99);

    std::vector<int> pair_eq = {42, 42};
    vqsort(std::span{pair_eq});
    EXPECT_EQ(pair_eq[0], 42);
    EXPECT_EQ(pair_eq[1], 42);
}

// 4. In-Register Bitonic Sorting Network Base Cases (N in [3, 64])
TEST(VQSortTest, InRegisterBaseCases) {
    std::mt19937 rng(42);
    for (std::size_t size = 3; size <= 64; ++size) {
        std::vector<int32_t> data(size);
        std::uniform_int_distribution<int32_t> dist(-1000, 1000);
        for (auto& x : data) {
            x = dist(rng);
        }
        auto expected = data;
        std::sort(expected.begin(), expected.end());

        vqsort(std::span{data});
        EXPECT_EQ(data, expected) << "Failed for size " << size;
    }
}

// 5. Partition Transition Boundary Sizes (N in [65, 512])
TEST(VQSortTest, PartitionTransitionBoundaries) {
    std::mt19937 rng(1337);
    const std::vector<std::size_t> test_sizes = {65,  66,  72,  80,  96,  127,
                                                 128, 129, 255, 256, 257, 512};
    for (std::size_t size : test_sizes) {
        std::vector<int32_t> data(size);
        std::uniform_int_distribution<int32_t> dist(-100000, 100000);
        for (auto& x : data) {
            x = dist(rng);
        }
        auto expected = data;
        std::sort(expected.begin(), expected.end());

        vqsort(std::span{data});
        EXPECT_EQ(data, expected) << "Failed for boundary size " << size;
    }
}

// 6. Distributions across N = 10,000 and 100,000
TEST(VQSortTest, StrictlySortedLinearPath) {
    std::vector<int32_t> data(10000);
    std::iota(data.begin(), data.end(), 1);
    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(VQSortTest, StrictlyReverseSortedLinearPath) {
    std::vector<int32_t> data(10000);
    std::iota(data.rbegin(), data.rend(), 1);
    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(VQSortTest, PresortedNinetyFivePercent) {
    std::vector<int32_t> data(10000);
    std::iota(data.begin(), data.end(), 0);
    std::mt19937 rng(999);
    for (std::size_t i = 0; i < 500; ++i) {
        std::swap(data[rng() % data.size()], data[rng() % data.size()]);
    }
    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(VQSortTest, SingleValueDuplicates) {
    std::vector<int32_t> data(10000, 42);
    vqsort(std::span{data});
    for (int x : data) {
        EXPECT_EQ(x, 42);
    }
}

TEST(VQSortTest, ManyDuplicatesFewUniqueKeys) {
    std::mt19937 rng(777);
    std::vector<int32_t> data(50000);
    std::uniform_int_distribution<int32_t> dist(1, 3);
    for (auto& x : data) {
        x = dist(rng);
    }
    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(VQSortTest, PipeOrganDistribution) {
    std::vector<int32_t> data(20000);
    std::size_t half = data.size() / 2;
    for (std::size_t i = 0; i < half; ++i) {
        data[i] = static_cast<int32_t>(i);
        data[half + i] = static_cast<int32_t>(half - i);
    }
    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(VQSortTest, SawtoothDistribution) {
    std::vector<int32_t> data(20000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int32_t>(i % 50);
    }
    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// 7. Typed Tests: 32-bit/64-bit int/uint, float, double
TEST(VQSortTest, Unsigned32Bit) {
    std::mt19937 rng(123);
    std::vector<uint32_t> data(10000);
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);
    for (auto& x : data)
        x = dist(rng);
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    vqsort(std::span{data});
    EXPECT_EQ(data, expected);
}

TEST(VQSortTest, Signed64Bit) {
    std::mt19937_64 rng(456);
    std::vector<int64_t> data(10000);
    std::uniform_int_distribution<int64_t> dist(-1000000000LL, 1000000000LL);
    for (auto& x : data)
        x = dist(rng);
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    vqsort(std::span{data});
    EXPECT_EQ(data, expected);
}

TEST(VQSortTest, Unsigned64Bit) {
    std::mt19937_64 rng(789);
    std::vector<uint64_t> data(10000);
    std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);
    for (auto& x : data)
        x = dist(rng);
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    vqsort(std::span{data});
    EXPECT_EQ(data, expected);
}

TEST(VQSortTest, FloatIEEE754) {
    std::mt19937 rng(321);
    std::vector<float> data(10000);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& x : data)
        x = dist(rng);
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    vqsort(std::span{data});
    EXPECT_EQ(data, expected);
}

TEST(VQSortTest, DoubleIEEE754) {
    std::mt19937 rng(654);
    std::vector<double> data(10000);
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);
    for (auto& x : data)
        x = dist(rng);
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    vqsort(std::span{data});
    EXPECT_EQ(data, expected);
}

// 8. Custom Comparator Descending
TEST(VQSortTest, CustomComparatorDescending) {
    std::vector<int32_t> data = {9, 2, 8, 3, 7, 4, 6, 5, 1};
    vqsort(std::span{data}, std::greater<int32_t>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int32_t>{}));

    std::mt19937 rng(111);
    std::vector<int32_t> large(5000);
    for (auto& x : large)
        x = static_cast<int32_t>(rng() % 10000);
    vqsort(std::span{large}, std::greater<int32_t>{});
    EXPECT_TRUE(std::is_sorted(large.begin(), large.end(), std::greater<int32_t>{}));
}

// 9. C++20 Ranges & Projections
struct Item {
    int key;
    std::string payload;
};

TEST(VQSortTest, Cpp20RangesAndProjection) {
    std::vector<Item> items = {{10, "ten"}, {2, "two"}, {8, "eight"}, {1, "one"}, {5, "five"}};
    vqsort(items, std::ranges::less{}, &Item::key);
    EXPECT_TRUE(std::is_sorted(items.begin(), items.end(),
                               [](const Item& a, const Item& b) { return a.key < b.key; }));
}

// 10. Non-Arithmetic Type Fallback (std::string)
TEST(VQSortTest, NonArithmeticTypesString) {
    std::vector<std::string> words = {"vectorized", "quicksort",    "highway", "avx512",
                                      "avx2",       "partitioning", "bitonic"};
    vqsort(std::span{words});
    EXPECT_TRUE(std::is_sorted(words.begin(), words.end()));
}

// 11. Large Scale Verification (10^6 and 10^7 elements)
TEST(VQSortTest, OneMillionElementsUniform) {
    std::mt19937 rng(2024);
    std::vector<int32_t> data(1000000);
    std::uniform_int_distribution<int32_t> dist(-10000000, 10000000);
    for (auto& x : data)
        x = dist(rng);

    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(VQSortTest, TenMillionElementsScale) {
    std::mt19937 rng(42);
    std::vector<int32_t> data(10000000);
    std::uniform_int_distribution<int32_t> dist(-10000000, 10000000);
    for (auto& x : data)
        x = dist(rng);

    vqsort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// 12. Adapter & SortVariant Static Dispatch
TEST(VQSortTest, AdapterSortVariantDispatch) {
    std::vector<int32_t> data = {50, 10, 40, 20, 30};
    VQSort algo;
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    SortVariant variant = VQSort{};
    std::vector<int32_t> data2 = {99, 11, 88, 22, 77};
    std::visit([&](const auto& s) { s.sort(std::span{data2}); }, variant);
    EXPECT_TRUE(std::is_sorted(data2.begin(), data2.end()));
}

} // namespace
