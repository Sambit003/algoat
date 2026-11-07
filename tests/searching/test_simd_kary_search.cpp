#include "algoat/searching/searching.hpp"
#include "algoat/searching/simd_kary_search.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using namespace algoat::searching;

// Verify SearchAlgorithm concept compliance
static_assert(SearchAlgorithm<SIMDKarySearch, int>);
static_assert(SearchAlgorithm<SIMDKarySearch, float>);
static_assert(SearchAlgorithm<SIMDKarySearch, uint32_t>);

TEST(SIMDKarySearchTest, SearchAlgorithmConcept) {
    SIMDKarySearch algo;
    EXPECT_EQ(algo.name(), "simdkarysearch");
    EXPECT_TRUE(algo.requires_sorted());

    std::vector<int> data = {1, 3, 5, 7, 9};
    auto res_found = algo.search(std::span<int>{data}, 5);
    ASSERT_TRUE(res_found.has_value());
    EXPECT_EQ(res_found.value(), 2);

    auto res_missing = algo.search(std::span<const int>{data}, 6);
    EXPECT_FALSE(res_missing.has_value());
}

TEST(SIMDKarySearchTest, EmptySpan) {
    std::vector<int> data;
    EXPECT_EQ(simd_kary_search(std::span<const int>{data}, 42), 0);

    SIMDKarySearchTree<int, 8> tree8(data);
    EXPECT_EQ(tree8.size(), 0);
    EXPECT_EQ(tree8.lower_bound(42), 0);
    EXPECT_FALSE(tree8.find(42).has_value());

    SIMDKarySearchTree<int, 16> tree16(data);
    EXPECT_EQ(tree16.size(), 0);
    EXPECT_EQ(tree16.lower_bound(42), 0);
    EXPECT_FALSE(tree16.find(42).has_value());
}

TEST(SIMDKarySearchTest, SingleElementSpan) {
    std::vector<int> data = {42};
    EXPECT_EQ(simd_kary_search(std::span<const int>{data}, 10), 0);
    EXPECT_EQ(simd_kary_search(std::span<const int>{data}, 42), 0);
    EXPECT_EQ(simd_kary_search(std::span<const int>{data}, 50), 1);

    SIMDKarySearchTree<int, 8> tree(data);
    EXPECT_EQ(tree.size(), 1);
    EXPECT_EQ(tree.lower_bound(10), 0);
    EXPECT_EQ(tree.lower_bound(42), 0);
    EXPECT_EQ(tree.lower_bound(50), 1);

    EXPECT_FALSE(tree.find(10).has_value());
    ASSERT_TRUE(tree.find(42).has_value());
    EXPECT_EQ(tree.find(42).value(), 0);
    EXPECT_FALSE(tree.find(50).has_value());
}

TEST(SIMDKarySearchTest, SmallArraySearch) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70};
    std::span<const int> s{data};

    // Exact matches
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(simd_kary_search(s, data[i]), i) << "Failed at exact index " << i;
    }

    // Interleaved values
    EXPECT_EQ(simd_kary_search(s, 5), 0);
    EXPECT_EQ(simd_kary_search(s, 15), 1);
    EXPECT_EQ(simd_kary_search(s, 25), 2);
    EXPECT_EQ(simd_kary_search(s, 35), 3);
    EXPECT_EQ(simd_kary_search(s, 65), 6);
    EXPECT_EQ(simd_kary_search(s, 75), 7);

    SIMDKarySearchTree<int, 8> tree(data);
    EXPECT_EQ(tree.size(), 7);
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(tree.lower_bound(data[i]), i);
        auto f = tree.find(data[i]);
        ASSERT_TRUE(f.has_value());
        EXPECT_EQ(f.value(), i);
    }
    EXPECT_EQ(tree.lower_bound(5), 0);
    EXPECT_EQ(tree.lower_bound(15), 1);
    EXPECT_EQ(tree.lower_bound(25), 2);
    EXPECT_EQ(tree.lower_bound(65), 6);
    EXPECT_EQ(tree.lower_bound(75), 7);
    EXPECT_FALSE(tree.find(15).has_value());
    EXPECT_FALSE(tree.find(75).has_value());
}

TEST(SIMDKarySearchTest, BoundariesAroundSimdWidths) {
    for (std::size_t n : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128}) {
        std::vector<int> data(n);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = static_cast<int>(i * 10);
        }
        std::span<const int> s{data};
        SIMDKarySearchTree<int, 8> tree(data);

        for (int q = -5; q <= static_cast<int>(n * 10 + 5); ++q) {
            auto it = std::lower_bound(data.begin(), data.end(), q);
            std::size_t expected = static_cast<std::size_t>(it - data.begin());
            EXPECT_EQ(simd_kary_search(s, q), expected)
                << "simd_kary_search failed for n=" << n << ", q=" << q;
            EXPECT_EQ(tree.lower_bound(q), expected)
                << "SIMDKarySearchTree failed for n=" << n << ", q=" << q;
        }
    }
}

TEST(SIMDKarySearchTest, DuplicatesHandling) {
    std::vector<int> data = {1, 2, 2, 2, 3, 3, 5, 5, 5, 5, 7};
    std::span<const int> s{data};
    SIMDKarySearchTree<int, 8> tree(data);

    for (int q = 0; q <= 8; ++q) {
        auto it = std::lower_bound(data.begin(), data.end(), q);
        std::size_t expected = static_cast<std::size_t>(it - data.begin());
        EXPECT_EQ(simd_kary_search(s, q), expected) << "Failed for q = " << q;
        EXPECT_EQ(tree.lower_bound(q), expected) << "Tree failed for q = " << q;
    }
}

TEST(SIMDKarySearchTest, Unsigned32Bit) {
    std::vector<std::uint32_t> data = {10u,         100u,        1000u,      0x7FFFFFFFu,
                                       0x80000000u, 0x80000001u, 0xFFFFFFFFu};
    std::span<const std::uint32_t> s{data};
    SIMDKarySearchTree<std::uint32_t, 8> tree(data);

    for (auto val : data) {
        EXPECT_EQ(simd_kary_search(s, val), tree.lower_bound(val));
        auto it = std::lower_bound(data.begin(), data.end(), val);
        std::size_t expected = static_cast<std::size_t>(it - data.begin());
        EXPECT_EQ(simd_kary_search(s, val), expected);
        EXPECT_EQ(tree.lower_bound(val), expected);
    }
}

TEST(SIMDKarySearchTest, Int64AndUint64) {
    std::vector<std::int64_t> d64 = {-1000LL, -5LL, 0LL, 42LL, 10000000000LL};
    std::span<const std::int64_t> s64{d64};
    SIMDKarySearchTree<std::int64_t, 4> tree64(d64);

    for (auto val : d64) {
        auto it = std::lower_bound(d64.begin(), d64.end(), val);
        std::size_t expected = static_cast<std::size_t>(it - d64.begin());
        EXPECT_EQ(simd_kary_search(s64, val), expected);
        EXPECT_EQ(tree64.lower_bound(val), expected);
    }

    std::vector<std::uint64_t> u64 = {0ULL, 50ULL, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL};
    std::span<const std::uint64_t> su64{u64};
    SIMDKarySearchTree<std::uint64_t, 4> tree_u64(u64);

    for (auto val : u64) {
        auto it = std::lower_bound(u64.begin(), u64.end(), val);
        std::size_t expected = static_cast<std::size_t>(it - u64.begin());
        EXPECT_EQ(simd_kary_search(su64, val), expected);
        EXPECT_EQ(tree_u64.lower_bound(val), expected);
    }
}

TEST(SIMDKarySearchTest, FloatAndDouble) {
    std::vector<float> f_data = {-10.5f, -0.0f, 0.0f, 1.25f, 3.14159f, 100.0f};
    std::span<const float> sf{f_data};
    SIMDKarySearchTree<float, 8> tree_f(f_data);

    for (float q :
         {-20.0f, -10.5f, -5.0f, 0.0f, 1.0f, 1.25f, 2.0f, 3.14159f, 50.0f, 100.0f, 200.0f}) {
        auto it = std::lower_bound(f_data.begin(), f_data.end(), q);
        std::size_t expected = static_cast<std::size_t>(it - f_data.begin());
        EXPECT_EQ(simd_kary_search(sf, q), expected) << "Float failed for q = " << q;
        EXPECT_EQ(tree_f.lower_bound(q), expected) << "Tree float failed for q = " << q;
    }

    std::vector<double> d_data = {-100.0, -1.0, 0.0, 2.71828, 50.0};
    std::span<const double> sd{d_data};
    SIMDKarySearchTree<double, 4> tree_d(d_data);

    for (double q : {-200.0, -100.0, -50.0, 0.0, 2.0, 2.71828, 10.0, 50.0, 80.0}) {
        auto it = std::lower_bound(d_data.begin(), d_data.end(), q);
        std::size_t expected = static_cast<std::size_t>(it - d_data.begin());
        EXPECT_EQ(simd_kary_search(sd, q), expected) << "Double failed for q = " << q;
        EXPECT_EQ(tree_d.lower_bound(q), expected) << "Tree double failed for q = " << q;
    }
}

TEST(SIMDKarySearchTest, LargeArrayMatchesStdLowerBound) {
    constexpr std::size_t N = 5000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i * 3);
    }
    std::span<const int> s{data};
    SIMDKarySearchTree<int, 8> tree8(data);
    SIMDKarySearchTree<int, 16> tree16(data);

    for (int q = -10; q < static_cast<int>(N * 3 + 10); q += 7) {
        auto it = std::lower_bound(data.begin(), data.end(), q);
        std::size_t expected = static_cast<std::size_t>(it - data.begin());
        EXPECT_EQ(simd_kary_search(s, q), expected) << "Failed for q = " << q;
        EXPECT_EQ(tree8.lower_bound(q), expected) << "Tree8 failed for q = " << q;
        EXPECT_EQ(tree16.lower_bound(q), expected) << "Tree16 failed for q = " << q;
    }
}

TEST(SIMDKarySearchTest, TenMillionQueriesEquivalence) {
    constexpr std::size_t N = 1000;
    constexpr std::size_t NUM_QUERIES_PER_TYPE = 5'000'000; // 5M int + 5M float = 10M total

    // 1. Integer verification (5M queries)
    std::vector<int32_t> int_data(N);
    for (std::size_t i = 0; i < N; ++i) {
        int_data[i] = static_cast<int32_t>(i * 4);
    }
    std::span<const int32_t> int_span{int_data};
    SIMDKarySearchTree<int32_t, 8> tree_int(int_data);

    // Fast Xorshift32 PRNG for deterministic, high-throughput verification
    uint32_t state = 0x12345678;
    auto xorshift32 = [&state]() noexcept -> uint32_t {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };

    for (std::size_t q = 0; q < NUM_QUERIES_PER_TYPE; ++q) {
        int32_t key = static_cast<int32_t>(xorshift32() % (N * 4 + 20)) - 10;
        auto it = std::lower_bound(int_data.begin(), int_data.end(), key);
        std::size_t expected = static_cast<std::size_t>(it - int_data.begin());

        ASSERT_EQ(simd_kary_search(int_span, key), expected);
        ASSERT_EQ(tree_int.lower_bound(key), expected);
    }

    // 2. IEEE-754 Float verification (5M queries)
    std::vector<float> float_data(N);
    for (std::size_t i = 0; i < N; ++i) {
        float_data[i] = static_cast<float>(i * 2.5f);
    }
    std::span<const float> float_span{float_data};
    SIMDKarySearchTree<float, 8> tree_float(float_data);

    for (std::size_t q = 0; q < NUM_QUERIES_PER_TYPE; ++q) {
        float key = static_cast<float>(xorshift32() % (N * 30)) * 0.1f - 5.0f;
        auto it = std::lower_bound(float_data.begin(), float_data.end(), key);
        std::size_t expected = static_cast<std::size_t>(it - float_data.begin());

        ASSERT_EQ(simd_kary_search(float_span, key), expected);
        ASSERT_EQ(tree_float.lower_bound(key), expected);
    }
}
