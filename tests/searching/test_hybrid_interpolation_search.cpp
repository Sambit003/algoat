#include "algoat/core/dispatcher.hpp"
#include "algoat/searching/hybrid_interpolation_search.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace algoat::searching;

class HybridInterpolationSearchTest : public ::testing::Test {
protected:
    HybridInterpolationSearch algo;
};

TEST_F(HybridInterpolationSearchTest, EmptyInput) {
    std::vector<int> data;
    auto res = algo.search(std::span{data}, 42);
    EXPECT_FALSE(res.has_value());

    auto free_res = hybrid_interpolation_search(std::span<const int>{data}, 42);
    EXPECT_FALSE(free_res.has_value());
}

TEST_F(HybridInterpolationSearchTest, SingleElement) {
    std::vector<int> data = {42};
    auto res_found = algo.search(std::span{data}, 42);
    ASSERT_TRUE(res_found.has_value());
    EXPECT_EQ(res_found.value(), 0);

    auto res_missing = algo.search(std::span{data}, 10);
    EXPECT_FALSE(res_missing.has_value());
}

TEST_F(HybridInterpolationSearchTest, SmallArrayBranchlessScan) {
    for (int size = 1; size <= 16; ++size) {
        std::vector<int> data(size);
        for (int i = 0; i < size; ++i) {
            data[i] = i * 10;
        }

        // Test every element present
        for (int i = 0; i < size; ++i) {
            auto res = algo.search(std::span{data}, i * 10);
            ASSERT_TRUE(res.has_value()) << "Failed for size=" << size << ", target=" << i * 10;
            EXPECT_EQ(res.value(), static_cast<std::size_t>(i));
        }

        // Test missing elements (smaller, in-between, larger)
        EXPECT_FALSE(algo.search(std::span{data}, -5).has_value());
        EXPECT_FALSE(algo.search(std::span{data}, 5).has_value());
        EXPECT_FALSE(algo.search(std::span{data}, size * 10).has_value());
    }
}

TEST_F(HybridInterpolationSearchTest, PrecisionSafetyLarge64BitIntegers) {
    // Values strictly greater than 2^53 (which would truncate in IEEE-754 double mantissa)
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
        auto res = algo.search(std::span{data}, data[i]);
        ASSERT_TRUE(res.has_value()) << "Failed finding large uint64 at index " << i;
        EXPECT_EQ(res.value(), i);
    }

    // Target between base and base + 1 (not present)
    auto missing = algo.search(std::span{data}, base + 50ULL);
    EXPECT_FALSE(missing.has_value());
}

TEST_F(HybridInterpolationSearchTest, Signed64BitExtremeSpan) {
    // Span covering the entire int64 range from MIN to MAX
    std::vector<int64_t> data = {std::numeric_limits<int64_t>::min(),
                                 std::numeric_limits<int64_t>::min() + 10,
                                 -1000LL,
                                 0LL,
                                 1000LL,
                                 std::numeric_limits<int64_t>::max() - 10,
                                 std::numeric_limits<int64_t>::max()};

    for (std::size_t i = 0; i < data.size(); ++i) {
        auto res = algo.search(std::span{data}, data[i]);
        ASSERT_TRUE(res.has_value()) << "Failed finding int64 at index " << i;
        EXPECT_EQ(res.value(), i);
    }

    EXPECT_FALSE(algo.search(std::span{data}, -999LL).has_value());
    EXPECT_FALSE(algo.search(std::span{data}, 500LL).has_value());
}

TEST_F(HybridInterpolationSearchTest, AdversarialClusteringTermination) {
    // 10^6 elements with severe exponential skew / clustering at 0
    constexpr std::size_t N = 1000000;
    std::vector<uint64_t> data(N, 0ULL);
    data[N - 1] = 1000000000ULL;

    // Searching for the outlier at the end
    auto res_outlier = algo.search(std::span{data}, 1000000000ULL);
    ASSERT_TRUE(res_outlier.has_value());
    EXPECT_EQ(res_outlier.value(), N - 1);

    // Searching for 0 (duplicate cluster)
    auto res_zero = algo.search(std::span{data}, 0ULL);
    ASSERT_TRUE(res_zero.has_value());
    EXPECT_EQ(data[res_zero.value()], 0ULL);

    // Searching for a non-existent element in clustered data
    // Standard interpolation search would degrade to O(N) linear scan, taking 10^6 iterations.
    // Hybrid search must adaptively fallback to binary search and terminate in O(log N) iterations.
    auto res_missing = algo.search(std::span{data}, 1ULL);
    EXPECT_FALSE(res_missing.has_value());

    auto res_missing_high = algo.search(std::span{data}, 500000000ULL);
    EXPECT_FALSE(res_missing_high.has_value());
}

TEST_F(HybridInterpolationSearchTest, ExponentialGrowthDistribution) {
    constexpr std::size_t N = 100000;
    std::vector<uint64_t> data(N);
    // Severe cubic distribution causing non-linear probe drift
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<uint64_t>(i) * i * i;
    }

    for (std::size_t i = 0; i < N; i += 2345) {
        auto res = algo.search(std::span{data}, data[i]);
        ASSERT_TRUE(res.has_value());
        EXPECT_EQ(res.value(), i);
    }

    // Value between cubes
    EXPECT_FALSE(algo.search(std::span{data}, data[50] + 1).has_value());
}

TEST_F(HybridInterpolationSearchTest, UniformlyDistributedLarge) {
    constexpr std::size_t N = 100000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i * 3);
    }

    for (std::size_t i = 0; i < N; i += 1234) {
        auto res = algo.search(std::span{data}, static_cast<int>(i * 3));
        ASSERT_TRUE(res.has_value());
        EXPECT_EQ(res.value(), i);
    }

    EXPECT_FALSE(algo.search(std::span{data}, -1).has_value());
    EXPECT_FALSE(algo.search(std::span{data}, 1).has_value());
    EXPECT_FALSE(algo.search(std::span{data}, static_cast<int>(N * 3)).has_value());
}

TEST_F(HybridInterpolationSearchTest, AllIdenticalElements) {
    std::vector<int> data(500, 7);
    auto res = algo.search(std::span{data}, 7);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(data[res.value()], 7);

    EXPECT_FALSE(algo.search(std::span{data}, 6).has_value());
    EXPECT_FALSE(algo.search(std::span{data}, 8).has_value());
}

TEST_F(HybridInterpolationSearchTest, NonArithmeticFallback) {
    std::vector<std::string> words = {"alpha",   "bravo", "charlie", "delta", "echo",
                                      "foxtrot", "golf",  "hotel",   "india", "juliet"};

    auto res = algo.search(std::span{words}, std::string("echo"));
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 4);

    EXPECT_FALSE(algo.search(std::span{words}, std::string("zulu")).has_value());
}

TEST_F(HybridInterpolationSearchTest, DispatcherIntegration) {
    algoat::core::AlgoConfig config;
    config.searching.prefer = "hybridinterpolationsearch";
    algoat::core::Dispatcher dispatcher(config);

    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80};
    auto res = dispatcher.search(std::span{data}, 60);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
}
