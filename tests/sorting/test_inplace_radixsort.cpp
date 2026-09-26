#include "algoat/sorting/radixsort.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

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

using namespace algoat::sorting;

template <typename T> void verify_inplace_radix_sort(std::vector<T>& data) {
    auto expected = data;
    std::sort(expected.begin(), expected.end());
    inplace_radix_sort(std::span{data});
    EXPECT_EQ(data, expected);
}

TEST(InPlaceRadixSortTest, ZeroHeapAllocations) {
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

    inplace_radix_sort(std::span{data});

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
    g_track_allocations.store(false, std::memory_order_seq_cst);

    EXPECT_EQ(g_active_allocations.load(std::memory_order_relaxed), 0)
        << "inplace_radix_sort must perform strictly zero dynamic heap allocations!";
#endif
    EXPECT_EQ(data, expected);
}

TEST(InPlaceRadixSortTest, Sparse64BitIntegerRange) {
    std::vector<int64_t> data = {
        0,  std::numeric_limits<int64_t>::max(),     std::numeric_limits<int64_t>::min(),     -1,
        1,  std::numeric_limits<int64_t>::max() - 1, std::numeric_limits<int64_t>::min() + 1, 42,
        -42};

    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, Sparse32BitIntegerRange) {
    std::vector<int32_t> data = {
        0,   std::numeric_limits<int32_t>::max(),     std::numeric_limits<int32_t>::min(),    -100,
        100, std::numeric_limits<int32_t>::max() - 1, std::numeric_limits<int32_t>::min() + 1};

    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, UnsignedIntegers64Bit) {
    std::vector<uint64_t> data = {
        0ULL, std::numeric_limits<uint64_t>::max(),     1ULL << 63,        (1ULL << 63) - 1,
        1ULL, std::numeric_limits<uint64_t>::max() - 1, 123456789012345ULL};

    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, IEEE754Floats) {
    std::vector<float> data = {3.14f,
                               -2.71f,
                               0.0f,
                               -0.0f,
                               100.5f,
                               -100.5f,
                               0.0001f,
                               -0.0001f,
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()};

    inplace_radix_sort(std::span{data});

    for (std::size_t i = 1; i < data.size(); ++i) {
        // -0.0f and 0.0f compare equal with == or <=
        EXPECT_TRUE(data[i - 1] <= data[i] ||
                    (std::signbit(data[i - 1]) && !std::signbit(data[i])));
    }
    EXPECT_TRUE(std::isinf(data.front()) && data.front() < 0.0f);
    EXPECT_TRUE(std::isinf(data.back()) && data.back() > 0.0f);
}

TEST(InPlaceRadixSortTest, IEEE754DoublesLarge) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    std::vector<double> data(5000);
    for (auto& x : data) {
        x = dist(gen);
    }
    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, CustomStructProjection) {
    struct Player {
        std::string name;
        int score;
        double rank;
    };

    std::vector<Player> players = {{"Alice", 95, 1.2},
                                   {"Bob", 72, 3.4},
                                   {"Charlie", -10, 0.5},
                                   {"Diana", 100, 9.9},
                                   {"Evan", 0, -2.1}};

    // Sort by integer projection
    inplace_radix_sort(std::span{players}, [](const Player& p) { return p.score; });
    EXPECT_TRUE(std::is_sorted(players.begin(), players.end(),
                               [](const Player& a, const Player& b) { return a.score < b.score; }));

    // Sort by double projection
    inplace_radix_sort(std::span{players}, [](const Player& p) { return p.rank; });
    EXPECT_TRUE(std::is_sorted(players.begin(), players.end(),
                               [](const Player& a, const Player& b) { return a.rank < b.rank; }));
}

TEST(InPlaceRadixSortTest, LargeRandomInt32) {
    std::mt19937 gen(999);
    std::uniform_int_distribution<int32_t> dist(-100000000, 100000000);
    std::vector<int32_t> data(100000);
    for (auto& x : data) {
        x = dist(gen);
    }
    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, LargeRandomInt64) {
    std::mt19937_64 gen(777);
    std::uniform_int_distribution<int64_t> dist(-1000000000000LL, 1000000000000LL);
    std::vector<int64_t> data(100000);
    for (auto& x : data) {
        x = dist(gen);
    }
    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, CommonPrefixSkipping) {
    // 50,000 integers with identical 3 high bytes
    std::mt19937 gen(123);
    std::uniform_int_distribution<int32_t> dist(0, 255);
    std::vector<int32_t> data(50000);
    constexpr int32_t prefix = 0x12345600;
    for (auto& x : data) {
        x = prefix | dist(gen);
    }
    verify_inplace_radix_sort(data);
}

TEST(InPlaceRadixSortTest, ExhaustiveSizes0To150) {
    std::mt19937 gen(55);
    for (std::size_t size = 0; size <= 150; ++size) {
        std::vector<int> data(size);
        for (std::size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int>(size - i);
        }
        inplace_radix_sort(std::span{data});
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end())) << "Failed on size " << size;
    }
}
