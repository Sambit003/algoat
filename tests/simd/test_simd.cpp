#include "algoat/simd/isa.hpp"
#include "algoat/simd/partition.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

TEST(SimdIsaTests, DetectIsaReturnsValidEnum) {
    auto isa = algoat::simd::detect_host_isa();
    // It should just not crash and return a valid enum
    EXPECT_GE(static_cast<int>(isa), 0);
    EXPECT_LE(static_cast<int>(isa), 3);
}

TEST(SimdPartitionTests, BasicPartitionInt32) {
    std::vector<int32_t> data = {9, 2, 7, 1, 8, 5, 3, 6, 4};
    int32_t pivot = 5;

    // Size is 9, so we pass 9 elements and a pivot
    std::size_t split = algoat::simd::partition_simd(data.data(), data.size(), pivot);

    // Everything before split should be < 5
    for (std::size_t i = 0; i < split; ++i) {
        EXPECT_LT(data[i], pivot);
    }
    // Everything from split onwards should be >= 5
    for (std::size_t i = split; i < data.size(); ++i) {
        EXPECT_GE(data[i], pivot);
    }
}

TEST(SimdPartitionTests, LargePartitionInt64) {
    std::vector<int64_t> data(1000);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end()); // 999 down to 0

    int64_t pivot = 500;
    std::size_t split = algoat::simd::partition_simd(data.data(), data.size(), pivot);

    for (std::size_t i = 0; i < split; ++i) {
        EXPECT_LT(data[i], pivot);
    }
    for (std::size_t i = split; i < data.size(); ++i) {
        EXPECT_GE(data[i], pivot);
    }
    EXPECT_EQ(split, 500); // 0 to 499 are less than 500
}
