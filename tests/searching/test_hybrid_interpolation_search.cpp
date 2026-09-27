#include "algoat/searching/hybrid_interpolation_search.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace algoat::searching;

class HybridInterpolationSearchTest : public ::testing::Test {};

TEST_F(HybridInterpolationSearchTest, EmptyInput) {
    std::vector<int> data;
    auto free_res = hybrid_interpolation_search(std::span<const int>{data}, 42);
    EXPECT_FALSE(free_res.has_value());
}

TEST_F(HybridInterpolationSearchTest, SingleElement) {
    std::vector<int> data = {42};
    auto res_found = hybrid_interpolation_search(std::span<const int>{data}, 42);
    ASSERT_TRUE(res_found.has_value());
    EXPECT_EQ(res_found.value(), 0);

    auto res_missing = hybrid_interpolation_search(std::span<const int>{data}, 10);
    EXPECT_FALSE(res_missing.has_value());
}

TEST_F(HybridInterpolationSearchTest, SmallArrayBranchlessScan) {
    for (int size = 1; size <= 16; ++size) {
        std::vector<int> data(size);
        for (int i = 0; i < size; ++i) {
            data[i] = i * 10;
        }

        for (int i = 0; i < size; ++i) {
            auto res = hybrid_interpolation_search(std::span<const int>{data}, i * 10);
            ASSERT_TRUE(res.has_value()) << "Failed for size=" << size << ", target=" << i * 10;
            EXPECT_EQ(res.value(), static_cast<std::size_t>(i));
        }

        EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, -5).has_value());
        EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, 5).has_value());
        EXPECT_FALSE(
            hybrid_interpolation_search(std::span<const int>{data}, size * 10).has_value());
    }
}

TEST_F(HybridInterpolationSearchTest, PrecisionSafetyLarge64BitIntegers) {
    constexpr uint64_t base = 1ULL << 55;
    std::vector<uint64_t> data = {0ULL,
                                  base,
                                  base + 1ULL,
                                  base + 2ULL,
                                  base + 100ULL,
                                  base + (1ULL << 20),
                                  std::numeric_limits<uint64_t>::max() - 1ULL,
                                  std::numeric_limits<uint64_t>::max()};

    for (std::size_t i = 0; i < data.size(); ++i) {
        auto res = hybrid_interpolation_search(std::span<const uint64_t>{data}, data[i]);
        ASSERT_TRUE(res.has_value()) << "Failed finding large uint64 at index " << i;
        EXPECT_EQ(res.value(), i);
    }

    auto missing = hybrid_interpolation_search(std::span<const uint64_t>{data},
                                               static_cast<uint64_t>(base + 50ULL));
    EXPECT_FALSE(missing.has_value());
}

TEST_F(HybridInterpolationSearchTest, Signed64BitExtremeSpan) {
    std::vector<int64_t> data = {std::numeric_limits<int64_t>::min(),
                                 std::numeric_limits<int64_t>::min() + 10,
                                 -1000LL,
                                 0LL,
                                 1000LL,
                                 std::numeric_limits<int64_t>::max() - 10,
                                 std::numeric_limits<int64_t>::max()};

    for (std::size_t i = 0; i < data.size(); ++i) {
        auto res = hybrid_interpolation_search(std::span<const int64_t>{data}, data[i]);
        ASSERT_TRUE(res.has_value()) << "Failed finding int64 at index " << i;
        EXPECT_EQ(res.value(), i);
    }

    EXPECT_FALSE(
        hybrid_interpolation_search(std::span<const int64_t>{data}, static_cast<int64_t>(-999LL))
            .has_value());
    EXPECT_FALSE(
        hybrid_interpolation_search(std::span<const int64_t>{data}, static_cast<int64_t>(500LL))
            .has_value());
}

TEST_F(HybridInterpolationSearchTest, AdversarialClusteringTermination) {
    constexpr std::size_t N = 1000000;
    std::vector<uint64_t> data(N, 0ULL);
    data[N - 1] = 1000000000ULL;

    auto res_outlier = hybrid_interpolation_search(std::span<const uint64_t>{data},
                                                   static_cast<uint64_t>(1000000000ULL));
    ASSERT_TRUE(res_outlier.has_value());
    EXPECT_EQ(res_outlier.value(), N - 1);

    auto res_zero =
        hybrid_interpolation_search(std::span<const uint64_t>{data}, static_cast<uint64_t>(0ULL));
    ASSERT_TRUE(res_zero.has_value());
    EXPECT_EQ(data[res_zero.value()], 0ULL);

    auto res_missing =
        hybrid_interpolation_search(std::span<const uint64_t>{data}, static_cast<uint64_t>(1ULL));
    EXPECT_FALSE(res_missing.has_value());

    auto res_missing_high = hybrid_interpolation_search(std::span<const uint64_t>{data},
                                                        static_cast<uint64_t>(500000000ULL));
    EXPECT_FALSE(res_missing_high.has_value());
}

TEST_F(HybridInterpolationSearchTest, ExponentialGrowthDistribution) {
    constexpr std::size_t N = 100000;
    std::vector<uint64_t> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<uint64_t>(i) * i * i;
    }

    for (std::size_t i = 0; i < N; i += 2345) {
        auto res = hybrid_interpolation_search(std::span<const uint64_t>{data}, data[i]);
        ASSERT_TRUE(res.has_value());
        EXPECT_EQ(res.value(), i);
    }

    EXPECT_FALSE(hybrid_interpolation_search(std::span<const uint64_t>{data},
                                             static_cast<uint64_t>(data[50] + 1ULL))
                     .has_value());
}

TEST_F(HybridInterpolationSearchTest, UniformlyDistributedLarge) {
    constexpr std::size_t N = 100000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i * 3);
    }

    for (std::size_t i = 0; i < N; i += 1234) {
        auto res = hybrid_interpolation_search(std::span<const int>{data}, static_cast<int>(i * 3));
        ASSERT_TRUE(res.has_value());
        EXPECT_EQ(res.value(), i);
    }

    EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, -1).has_value());
    EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, 1).has_value());
    EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, static_cast<int>(N * 3))
                     .has_value());
}

TEST_F(HybridInterpolationSearchTest, AllIdenticalElements) {
    std::vector<int> data(500, 7);
    auto res = hybrid_interpolation_search(std::span<const int>{data}, 7);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(data[res.value()], 7);

    EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, 6).has_value());
    EXPECT_FALSE(hybrid_interpolation_search(std::span<const int>{data}, 8).has_value());
}

TEST_F(HybridInterpolationSearchTest, PointerFallbackAndInterpolation) {
    int arr[10];
    std::vector<int*> data;
    for (int i = 0; i < 10; ++i) {
        data.push_back(&arr[i]);
    }

    auto res = hybrid_interpolation_search(std::span<int* const>{data}, &arr[5]);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
}

enum class Timestamp : uint64_t { START = 0, MIDDLE = 1000, END = 2000 };

TEST_F(HybridInterpolationSearchTest, EnumInterpolation) {
    std::vector<Timestamp> data = {Timestamp::START, Timestamp::MIDDLE, Timestamp::END};
    auto res = hybrid_interpolation_search(std::span<const Timestamp>{data}, Timestamp::MIDDLE);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 1);
}
