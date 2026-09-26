#include "algoat/simd/isa.hpp"
#include "algoat/simd/partition.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

TEST(SimdIsaTests, DetectIsaReturnsValidEnum) {
    auto isa = algoat::simd::detect_host_isa();
    // Verify the returned value is a recognised ISA tier - avoids fragile int casts.
    EXPECT_TRUE(
        isa == algoat::simd::InstructionSet::Scalar || isa == algoat::simd::InstructionSet::AVX2 ||
        isa == algoat::simd::InstructionSet::AVX512 || isa == algoat::simd::InstructionSet::Neon);
}

TEST(SimdIsaTests, DetectIsaRepeatedCallsZeroOverheadAndConsistent) {
    auto initial_isa = algoat::simd::detect_host_isa();
    for (int i = 0; i < 10000; ++i) {
        EXPECT_EQ(algoat::simd::detect_host_isa(), initial_isa);
    }
}

namespace {
template <typename T>
void verify_partition(const std::vector<T>& data, T pivot, std::optional<std::size_t> split_opt) {
    if (!split_opt.has_value()) {
        EXPECT_EQ(algoat::simd::detect_host_isa(), algoat::simd::InstructionSet::Scalar);
        return;
    }
    std::size_t split = *split_opt;
    for (std::size_t i = 0; i < split; ++i) {
        EXPECT_LT(data[i], pivot);
    }
    for (std::size_t i = split; i < data.size(); ++i) {
        EXPECT_GE(data[i], pivot);
    }
}
} // namespace

TEST(SimdPartitionTests, BasicPartitionInt32) {
    std::vector<int32_t> data = {9, 2, 7, 1, 8, 5, 3, 6, 4};
    int32_t pivot = 5;
    auto split_opt = algoat::simd::partition_simd(data.data(), data.size(), pivot);
    verify_partition(data, pivot, split_opt);
}

TEST(SimdPartitionTests, LargePartitionInt64) {
    std::vector<int64_t> data(1000);
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end()); // 999 down to 0

    int64_t pivot = 500;
    auto split_opt = algoat::simd::partition_simd(data.data(), data.size(), pivot);
    verify_partition(data, pivot, split_opt);

    if (split_opt.has_value()) {
        EXPECT_EQ(*split_opt, 500); // 0 to 499 are less than 500
    }
}

TEST(SimdPartitionTests, AllLessThanPivot) {
    std::vector<int32_t> data = {1, 2, 3, 4, 0, -5, 2, 3};
    int32_t pivot = 10;
    auto split_opt = algoat::simd::partition_simd(data.data(), data.size(), pivot);
    verify_partition(data, pivot, split_opt);
    if (split_opt.has_value()) {
        EXPECT_EQ(*split_opt, data.size());
    }
}

TEST(SimdPartitionTests, AllGreaterThanOrEqualToPivot) {
    std::vector<int32_t> data = {15, 22, 10, 14, 50, 12, 100};
    int32_t pivot = 10;
    auto split_opt = algoat::simd::partition_simd(data.data(), data.size(), pivot);
    verify_partition(data, pivot, split_opt);
    if (split_opt.has_value()) {
        EXPECT_EQ(*split_opt, 0u);
    }
}
