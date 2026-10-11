#include "algoat/numerics/float16_sort.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::numerics;

TEST(Float16SortTest, EmptySpan) {
    std::vector<uint16_t> empty;
    sort_float16(std::span<uint16_t>{empty});
    EXPECT_TRUE(empty.empty());
}

TEST(Float16SortTest, SingleElement) {
    std::vector<uint16_t> single = {0x3C00}; // 1.0 in float16
    sort_float16(std::span<uint16_t>{single});
    EXPECT_EQ(single[0], 0x3C00);
}

TEST(Float16SortTest, SmallArrayOrdering) {
    // Tests known IEEE-754 float16 bit patterns
    // -inf: 0xFC00, -1.0: 0xBC00, -0.0: 0x8000, +0.0: 0x0000, 1.0: 0x3C00, +inf: 0x7C00
    std::vector<uint16_t> data = {0x3C00, 0x7C00, 0x8000, 0x0000, 0xFC00, 0xBC00};
    sort_float16(std::span<uint16_t>{data});

    std::vector<uint16_t> expected = {0xFC00, 0xBC00, 0x8000, 0x0000, 0x3C00, 0x7C00};
    EXPECT_EQ(data, expected);
}

TEST(Float16SortTest, MediumArrayRadixPass) {
    // 256 <= N < 65536 triggers the 2-pass 8-bit Radix Sort
    constexpr size_t N = 1000;
    std::vector<uint16_t> data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 65535);

    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<uint16_t>(dist(rng));
    }

    std::vector<uint16_t> expected = data;
    std::sort(expected.begin(), expected.end(), Float16Compare{});

    sort_float16(std::span<uint16_t>{data});
    EXPECT_EQ(data, expected);
}

TEST(Float16SortTest, LargeArrayCountingPass) {
    // N >= 65536 triggers the single-pass 16-bit Counting Sort
    constexpr size_t N = 70000;
    std::vector<uint16_t> data(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, 65535);

    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<uint16_t>(dist(rng));
    }

    std::vector<uint16_t> expected = data;
    std::sort(expected.begin(), expected.end(), Float16Compare{});

    sort_float16(std::span<uint16_t>{data});
    EXPECT_EQ(data, expected);
}

TEST(Float16SortTest, SignedInt16Coverage) {
    std::vector<int16_t> data = {10, -5, 0, 100, -200, 50};
    std::vector<int16_t> expected = data;
    std::sort(expected.begin(), expected.end(), Float16Compare{});

    sort_float16(std::span<int16_t>{data});
    EXPECT_EQ(data, expected);
}
