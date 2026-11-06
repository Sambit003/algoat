#include "algoat/sorting/ska_sort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
// Allocation tracking to verify O(1) auxiliary heap space guarantee in standard builds
static std::atomic<std::size_t> g_active_allocations{0};
static std::atomic<bool> g_track_allocations{false};

void* operator new(std::size_t size) {
    if (g_track_allocations.load(std::memory_order_relaxed)) {
        g_active_allocations.fetch_add(size, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
#endif

namespace algoat::sorting::testing {
namespace {

struct Item {
    int id;
    double priority;

    constexpr auto operator<=>(const Item&) const = default;
};

// 1. Concept conformance check
static_assert(SortAlgorithm<SkaSort, int>, "SkaSort must satisfy SortAlgorithm<int>");
static_assert(SortAlgorithm<SkaSort, double>, "SkaSort must satisfy SortAlgorithm<double>");

// 2. Compile-time constexpr check
constexpr bool test_constexpr_ska_sort() {
    std::array<int, 10> arr = {9, 1, 5, 3, 7, 2, 8, 4, 6, 0};
    ska_sort(arr);
    for (std::size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

static_assert(test_constexpr_ska_sort(), "ska_sort must be constexpr evaluatable");

template <typename T> void verify_ska_sort(std::vector<T>& data) {
    auto expected = data;
    std::sort(expected.begin(), expected.end());
    ska_sort(data);
    EXPECT_EQ(data, expected);
}

TEST(SkaSortTest, AdapterConceptProperties) {
    SkaSort algo;
    EXPECT_EQ(algo.name(), "ska_sort");
    EXPECT_EQ(algo.preferred_min_size(), 128u);

    std::vector<int> data = {5, 2, 8, 1, 9, 3};
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SkaSortTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    auto it_empty = ska_sort(empty);
    EXPECT_EQ(it_empty, empty.end());
    EXPECT_TRUE(empty.empty());

    ska_sort(std::span<int>{empty});
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    auto it_single = ska_sort(single);
    EXPECT_EQ(it_single, single.end());
    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);

    ska_sort(std::span<int>{single});
    EXPECT_EQ(single[0], 42);
}

TEST(SkaSortTest, TwoElements) {
    std::vector<int> sorted_pair = {10, 20};
    verify_ska_sort(sorted_pair);

    std::vector<int> reversed_pair = {20, 10};
    verify_ska_sort(reversed_pair);

    std::vector<int> equal_pair = {10, 10};
    verify_ska_sort(equal_pair);
}

TEST(SkaSortTest, ThresholdBoundarySizes) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (std::size_t size :
         {3, 10, 24, 64, 100, 127, 128, 129, 200, 255, 256, 257, 512, 1023, 1024, 1025, 2048}) {
        std::vector<int> data(size);
        for (auto& x : data) {
            x = dist(gen);
        }
        verify_ska_sort(data);
    }
}

TEST(SkaSortTest, PresortedAndDuplicatePatterns) {
    // Strictly sorted
    std::vector<int> sorted(500);
    std::iota(sorted.begin(), sorted.end(), 1);
    verify_ska_sort(sorted);

    // Strictly reversed
    std::vector<int> reversed = sorted;
    std::reverse(reversed.begin(), reversed.end());
    verify_ska_sort(reversed);

    // All duplicates
    std::vector<int> duplicates(1000, 42);
    verify_ska_sort(duplicates);

    // Few unique (3 values across 3000 items)
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> dist(1, 3);
    std::vector<int> few_unique(3000);
    for (auto& x : few_unique) {
        x = dist(gen);
    }
    verify_ska_sort(few_unique);
}

TEST(SkaSortTest, NumericTypesCoverage) {
    // 8-bit
    std::vector<int8_t> i8 = {0, 127, -128, -1, 1, -42, 42};
    verify_ska_sort(i8);
    std::vector<uint8_t> u8 = {0, 255, 128, 1, 42, 200};
    verify_ska_sort(u8);

    // 16-bit
    std::vector<int16_t> i16 = {0, 32767, -32768, -1, 1, 1000, -1000};
    verify_ska_sort(i16);
    std::vector<uint16_t> u16 = {0, 65535, 32768, 1, 1000, 50000};
    verify_ska_sort(u16);

    // 32-bit
    std::vector<int32_t> i32 = {0,
                                std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::min(),
                                -1,
                                1,
                                123456,
                                -123456};
    verify_ska_sort(i32);
    std::vector<uint32_t> u32 = {
        0, std::numeric_limits<uint32_t>::max(), 1U << 31, (1U << 31) - 1, 1, 999999};
    verify_ska_sort(u32);

    // 64-bit
    std::vector<int64_t> i64 = {0, std::numeric_limits<int64_t>::max(),
                                std::numeric_limits<int64_t>::min(), -1LL, 1LL};
    verify_ska_sort(i64);
    std::vector<uint64_t> u64 = {0ULL, std::numeric_limits<uint64_t>::max(), 1ULL << 63,
                                 (1ULL << 63) - 1, 1ULL};
    verify_ska_sort(u64);
}

TEST(SkaSortTest, FloatingPointHandling) {
    std::vector<float> f_data = {0.0f,
                                 -0.0f,
                                 1.0f,
                                 -1.0f,
                                 3.14159f,
                                 -2.71828f,
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::denorm_min(),
                                 -std::numeric_limits<float>::denorm_min(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::lowest()};

    auto expected = f_data;
    std::sort(expected.begin(), expected.end());
    ska_sort(f_data);

    // Verify sortedness
    for (std::size_t i = 1; i < f_data.size(); ++i) {
        EXPECT_TRUE(f_data[i - 1] <= f_data[i] || (f_data[i - 1] == 0.0f && f_data[i] == 0.0f));
    }

    // Verify -0.0f sorts before +0.0f in bitwise total ordering
    std::vector<float> zeros = {0.0f, -0.0f, 0.0f, -0.0f};
    ska_sort(zeros);
    EXPECT_TRUE(std::signbit(zeros[0]));
    EXPECT_TRUE(std::signbit(zeros[1]));
    EXPECT_FALSE(std::signbit(zeros[2]));
    EXPECT_FALSE(std::signbit(zeros[3]));

    // Doubles
    std::vector<double> d_data = {0.0,
                                  -0.0,
                                  1.0,
                                  -1.0,
                                  1e100,
                                  -1e100,
                                  std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity()};
    ska_sort(d_data);
    for (std::size_t i = 1; i < d_data.size(); ++i) {
        EXPECT_TRUE(d_data[i - 1] <= d_data[i] || (d_data[i - 1] == 0.0 && d_data[i] == 0.0));
    }
}

TEST(SkaSortTest, CommonPrefixStress) {
    // 50,000 numbers sharing identical top 3 bytes (0x12345600 | x)
    std::vector<uint32_t> data(50000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = 0x12345600u | (static_cast<uint32_t>(i) % 256u);
    }
    verify_ska_sort(data);
}

TEST(SkaSortTest, StructProjection) {
    std::vector<Item> items = {{5, 2.5}, {1, 9.1}, {9, 1.2}, {3, 7.8}, {2, 0.5}};

    // Sort by id projection
    ska_sort(items, &Item::id);
    for (std::size_t i = 1; i < items.size(); ++i) {
        EXPECT_LE(items[i - 1].id, items[i].id);
    }

    // Sort by priority projection (double)
    ska_sort(items, &Item::priority);
    for (std::size_t i = 1; i < items.size(); ++i) {
        EXPECT_LE(items[i - 1].priority, items[i].priority);
    }
}

TEST(SkaSortTest, TuplesAndPairs) {
    std::vector<std::pair<int, double>> pairs = {
        {10, 2.5}, {5, 1.0}, {10, 1.2}, {5, 0.5}, {1, 9.9}};
    auto expected_pairs = pairs;
    std::sort(expected_pairs.begin(), expected_pairs.end());
    ska_sort(pairs);
    EXPECT_EQ(pairs, expected_pairs);

    std::vector<std::tuple<int, int, int>> tuples = {
        {1, 2, 3}, {1, 1, 5}, {0, 9, 9}, {1, 1, 2}, {2, 0, 0}};
    auto expected_tuples = tuples;
    std::sort(expected_tuples.begin(), expected_tuples.end());
    ska_sort(tuples);
    EXPECT_EQ(tuples, expected_tuples);
}

TEST(SkaSortTest, FallbackNonRadixTypes) {
    // Strings are non-radix sortable scalars -> falls back to pdqsort
    std::vector<std::string> strings = {"banana", "apple", "cherry", "date", "fig", "apricot"};
    auto expected_strings = strings;
    std::sort(expected_strings.begin(), expected_strings.end());
    ska_sort(strings);
    EXPECT_EQ(strings, expected_strings);

    // Tuples containing non-radix elements
    std::vector<std::pair<int, std::string>> mixed = {
        {2, "beta"}, {1, "alpha"}, {2, "alpha"}, {1, "beta"}};
    auto expected_mixed = mixed;
    std::sort(expected_mixed.begin(), expected_mixed.end());
    ska_sort(mixed);
    EXPECT_EQ(mixed, expected_mixed);
}

TEST(SkaSortTest, Cpp20RangesInterface) {
    // 1. Borrowed range lvalue returns iterator to end
    std::vector<int> v = {4, 2, 7, 1, 9};
    auto it = ska_sort(v);
    EXPECT_EQ(it, v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));

    // 2. Rvalue container returns dangling
    static_assert(
        std::same_as<decltype(ska_sort(std::vector<int>{3, 1, 2})), std::ranges::dangling>,
        "ska_sort on rvalue non-borrowed range must return dangling");

    // 3. Span overload returns void
    static_assert(std::same_as<decltype(ska_sort(std::span<int>{v})), void>,
                  "ska_sort on std::span<T> must return void");

    // 4. Span with projection returns void
    static_assert(std::same_as<decltype(ska_sort(std::span<Item>{}, &Item::id)), void>,
                  "ska_sort on std::span<T, Proj> must return void");
}

TEST(SkaSortTest, ZeroHeapAllocations) {
    std::mt19937 gen(1337);
    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
    std::vector<int32_t> data(100000);
    for (auto& x : data) {
        x = dist(gen);
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
    g_active_allocations.store(0, std::memory_order_relaxed);
    g_track_allocations.store(true, std::memory_order_seq_cst);
#endif

    ska_sort(std::span{data});

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
    g_track_allocations.store(false, std::memory_order_seq_cst);

    EXPECT_EQ(g_active_allocations.load(std::memory_order_relaxed), 0)
        << "ska_sort must perform strictly zero dynamic heap allocations!";
#endif
    EXPECT_EQ(data, expected);
}

TEST(SkaSortTest, BitLevelParity100k) {
    std::mt19937 gen(99999);
    std::uniform_int_distribution<int32_t> dist(std::numeric_limits<int32_t>::min(),
                                                std::numeric_limits<int32_t>::max());
    std::vector<int32_t> data(100000);
    for (auto& x : data) {
        x = dist(gen);
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());
    ska_sort(data);
    EXPECT_EQ(data, expected);
}

TEST(SkaSortTest, NonContiguousRandomAccessRange) {
    std::deque<int> dq = {9, 1, 5, 3, 7, 2, 8, 4, 6, 0};
    ska_sort(dq);
    EXPECT_TRUE(std::is_sorted(dq.begin(), dq.end()));

    std::deque<int> large_dq(500);
    std::iota(large_dq.begin(), large_dq.end(), 1);
    std::reverse(large_dq.begin(), large_dq.end());
    ska_sort(large_dq);
    EXPECT_TRUE(std::is_sorted(large_dq.begin(), large_dq.end()));
}

TEST(SkaSortTest, RandomizedRuns100k) {
    std::mt19937 gen(424242);
    std::uniform_int_distribution<int> size_dist(0, 150);
    std::uniform_int_distribution<int> val_dist(-5000, 5000);

    for (int run = 0; run < 100000; ++run) {
        int n = size_dist(gen);
        std::vector<int> data(n);
        for (int& x : data) {
            x = val_dist(gen);
        }
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        ska_sort(data);
        ASSERT_EQ(data, expected);
    }
}

} // namespace
} // namespace algoat::sorting::testing
