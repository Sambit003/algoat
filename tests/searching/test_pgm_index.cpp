#include "algoat/searching/pgm_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace algoat::searching;

TEST(PGMIndexTest, EmptySpan) {
    std::vector<int> data;
    PGMIndex<int> index(std::span<const int>{data});
    EXPECT_EQ(index.segments_count(), 0);
    EXPECT_EQ(index.lower_bound(data, 42), 0);
    auto [lo, hi] = index.search_range(42);
    EXPECT_EQ(lo, 0);
    EXPECT_EQ(hi, 0);
    EXPECT_LE(hi - lo, 2 * PGMIndex<int>::epsilon);
}

TEST(PGMIndexTest, SingleElementSpan) {
    std::vector<int> data = {42};
    PGMIndex<int> index(data);
    EXPECT_GT(index.segments_count(), 0);

    // Queries: before, equal, after
    EXPECT_EQ(index.lower_bound(data, 10), 0);
    EXPECT_EQ(index.lower_bound(data, 42), 0);
    EXPECT_EQ(index.lower_bound(data, 50), 1);

    auto [lo, hi] = index.search_range(42);
    EXPECT_LE(hi - lo, 2 * PGMIndex<int>::epsilon);
    EXPECT_EQ(lo, 0);
    EXPECT_EQ(hi, 0);
}

TEST(PGMIndexTest, SmallArraySearch) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70};
    PGMIndex<int> index(data);

    // Exact hits
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(index.lower_bound(data, data[i]), i);
        auto [lo, hi] = index.search_range(data[i]);
        EXPECT_LE(lo, i);
        EXPECT_GE(hi, i);
        EXPECT_LE(hi - lo, 2 * PGMIndex<int>::epsilon);
    }

    // Values between elements
    EXPECT_EQ(index.lower_bound(data, 5), 0);
    EXPECT_EQ(index.lower_bound(data, 15), 1);
    EXPECT_EQ(index.lower_bound(data, 25), 2);
    EXPECT_EQ(index.lower_bound(data, 65), 6);
    EXPECT_EQ(index.lower_bound(data, 75), 7);
}

TEST(PGMIndexTest, AllElementsIdentical) {
    std::vector<int> data(100, 77);
    PGMIndex<int, 16, 4> index(data);

    EXPECT_EQ(index.lower_bound(data, 50), 0);
    EXPECT_EQ(index.lower_bound(data, 77), 0);
    EXPECT_EQ(index.lower_bound(data, 90), 100);

    auto [lo, hi] = index.search_range(77);
    EXPECT_LE(lo, 0);
    EXPECT_LE(hi - lo, 2 * 16);
}

TEST(PGMIndexTest, DuplicateKeysAndGaps) {
    std::vector<int> data = {10, 10, 10, 20, 20, 30, 40, 40, 40, 40, 50};
    PGMIndex<int, 4, 4> index(data);

    for (int target = 0; target <= 60; ++target) {
        auto expected = static_cast<std::size_t>(
            std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), target)));
        EXPECT_EQ(index.lower_bound(data, target), expected) << "Mismatch on target=" << target;
    }
}

TEST(PGMIndexTest, ExtremeKeyRangeWithDuplicates) {
    std::vector<int> data = {0, 0, 0, 0, 376594362, 0,           0, 0, 0, 0,
                             0, 0, 0, 0, 0,         -1770889286, 0, 0, 0, 0};
    std::sort(data.begin(), data.end());
    PGMIndex<int, 16, 4> index(data);

    int target = 376594362;
    auto expected = static_cast<std::size_t>(
        std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), target)));
    EXPECT_EQ(index.lower_bound(data, target), expected);
    EXPECT_EQ(index.lower_bound(data, -1770889286), 0);
    EXPECT_EQ(index.lower_bound(data, 0), 1);
    EXPECT_EQ(index.lower_bound(data, 1), 19);
}

TEST(PGMIndexTest, MaxKeyValueSentinelBoundary) {
    constexpr int64_t max_val = std::numeric_limits<int64_t>::max() - 100;
    std::vector<int64_t> data = {10, 100, 1000, max_val};
    PGMIndex<int64_t, 16, 4> index(data);

    // Query with numeric_limits::max()
    int64_t max_query = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(index.lower_bound(data, max_query), 4);
    EXPECT_EQ(index.lower_bound(data, max_val), 3);
    EXPECT_EQ(index.lower_bound(data, max_val + 1), 4);
}

TEST(PGMIndexTest, UniformDistributionParityAndErrorBounds) {
    constexpr std::size_t N = 10000;
    std::vector<int64_t> data(N);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(1, 1000000);

    for (std::size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    std::sort(data.begin(), data.end());

    PGMIndex<int64_t, 32, 8> index(data);

    // Provable error bound check: for all keys in data, search_range contains key
    for (std::size_t i = 0; i < N; ++i) {
        auto [lo, hi] = index.search_range(data[i]);
        EXPECT_LE(lo, i);
        EXPECT_GE(hi, i);
        EXPECT_LE(hi - lo, 2 * 32);
    }

    // Test 10,000 queries for exact parity with std::lower_bound
    std::uniform_int_distribution<int64_t> query_dist(0, 1000005);
    for (std::size_t q = 0; q < 10000; ++q) {
        int64_t target = query_dist(rng);
        auto expected = static_cast<std::size_t>(
            std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), target)));
        EXPECT_EQ(index.lower_bound(data, target), expected);
    }
}

TEST(PGMIndexTest, LogNormalDistributionParity) {
    constexpr std::size_t N = 20000;
    std::vector<int64_t> data;
    data.reserve(N);

    std::mt19937_64 rng(777);
    std::lognormal_distribution<double> log_dist(6.0, 1.5);

    for (std::size_t i = 0; i < N; ++i) {
        data.push_back(static_cast<int64_t>(log_dist(rng)));
    }
    std::sort(data.begin(), data.end());

    PGMIndex<int64_t, 64, 16> index(data);

    // Verify 5,000 queries
    std::uniform_int_distribution<int64_t> query_dist(0, data.back() + 100);
    for (std::size_t q = 0; q < 5000; ++q) {
        int64_t target = query_dist(rng);
        auto expected = static_cast<std::size_t>(
            std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), target)));
        EXPECT_EQ(index.lower_bound(data, target), expected);
    }
}

TEST(PGMIndexTest, ClusteredDistributionAndMemoryFootprint) {
    constexpr std::size_t N = 50000;
    std::vector<int64_t> data;
    data.reserve(N);

    std::mt19937_64 rng(1337);
    std::normal_distribution<double> cluster1(10000.0, 500.0);
    std::normal_distribution<double> cluster2(50000.0, 2000.0);

    for (std::size_t i = 0; i < N / 2; ++i) {
        data.push_back(static_cast<int64_t>(std::max(0.0, cluster1(rng))));
        data.push_back(static_cast<int64_t>(std::max(0.0, cluster2(rng))));
    }
    std::sort(data.begin(), data.end());

    PGMIndex<int64_t, 64, 16> index(data);

    // Verify memory footprint < 0.5% of raw data size
    std::size_t raw_bytes = data.size() * sizeof(int64_t);
    std::size_t index_bytes = index.size_in_bytes();
    EXPECT_LT(index_bytes, raw_bytes / 200) // index_bytes < 0.5%
        << "index_bytes: " << index_bytes << " raw_bytes: " << raw_bytes;

    // Verify exact lower_bound parity on 5000 random queries
    std::uniform_int_distribution<int64_t> query_dist(0, 70000);
    for (std::size_t q = 0; q < 5000; ++q) {
        int64_t target = query_dist(rng);
        auto expected = static_cast<std::size_t>(
            std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), target)));
        EXPECT_EQ(index.lower_bound(data, target), expected);
    }
}

TEST(PGMIndexTest, LargeScaleParityStress) {
    constexpr std::size_t N = 100000;
    std::vector<int64_t> data(N);
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int64_t> dist(1, 10000000);

    for (std::size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    std::sort(data.begin(), data.end());

    PGMIndex<int64_t, 64, 16> index(data);

    std::uniform_int_distribution<int64_t> query_dist(0, 10000050);
    for (std::size_t q = 0; q < 100000; ++q) {
        int64_t target = query_dist(rng);
        auto expected = static_cast<std::size_t>(
            std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), target)));
        EXPECT_EQ(index.lower_bound(data, target), expected);
    }
}

TEST(PGMIndexTest, FloatingPointKeys) {
    std::vector<double> data = {1.1, 2.5, 3.8, 7.2, 10.9, 15.4, 20.0};
    PGMIndex<double, 4, 4> index(data);

    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(index.lower_bound(data, data[i]), i);
    }
    EXPECT_EQ(index.lower_bound(data, 0.5), 0);
    EXPECT_EQ(index.lower_bound(data, 5.0), 3);
    EXPECT_EQ(index.lower_bound(data, 25.0), data.size());
}

TEST(PGMIndexTest, RebuildIndex) {
    std::vector<int> data1 = {10, 20, 30};
    PGMIndex<int, 4, 4> index(data1);
    EXPECT_EQ(index.lower_bound(data1, 20), 1);

    std::vector<int> data2 = {100, 200, 300, 400, 500};
    index.build(data2);
    EXPECT_EQ(index.lower_bound(data2, 300), 2);
    EXPECT_EQ(index.lower_bound(data2, 50), 0);
    EXPECT_EQ(index.lower_bound(data2, 600), 5);
}
