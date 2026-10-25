#include <algoat/c_api/algoat_c.h>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

TEST(AlgoatCTest, SortInt64SequentialAndParallel) {
    std::vector<int64_t> data = {42, -5, 100, 0, -100, 3, 7, 2, 99, -1};
    algoat_sort_int64(data.data(), data.size(), 0);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    std::mt19937_64 rng(1337);
    std::vector<int64_t> large(5000);
    for (auto& x : large) {
        x = static_cast<int64_t>(rng());
    }
    algoat_sort_int64(large.data(), large.size(), 1);
    EXPECT_TRUE(std::is_sorted(large.begin(), large.end()));
}

TEST(AlgoatCTest, SortInt32SequentialAndParallel) {
    std::vector<int32_t> data = {10, -3, 5, 0, 999, -500, 2};
    algoat_sort_int32(data.data(), data.size(), 0);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    std::vector<int32_t> large(3000);
    for (size_t i = 0; i < large.size(); ++i) {
        large[i] = static_cast<int32_t>(3000 - i);
    }
    algoat_sort_int32(large.data(), large.size(), 1);
    EXPECT_TRUE(std::is_sorted(large.begin(), large.end()));
}

TEST(AlgoatCTest, SortFloat64SequentialAndParallel) {
    std::vector<double> data = {3.14, -0.5, 2.718, 0.0, -100.2, 1e6, -1e6};
    algoat_sort_float64(data.data(), data.size(), 0);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));

    std::vector<double> large(2048);
    for (size_t i = 0; i < large.size(); ++i) {
        large[i] = static_cast<double>(2048 - i) * 1.5;
    }
    algoat_sort_float64(large.data(), large.size(), 1);
    EXPECT_TRUE(std::is_sorted(large.begin(), large.end()));
}

TEST(AlgoatCTest, RadixSortInt64AndInt32) {
    std::vector<int64_t> data64 = {500, -20, 100000, 0, -99999, 42, 13};
    algoat_radix_sort_int64(data64.data(), data64.size());
    EXPECT_TRUE(std::is_sorted(data64.begin(), data64.end()));

    std::vector<int32_t> data32 = {10, -5, 1000, 0, -500, 77, 2};
    algoat_radix_sort_int32(data32.data(), data32.size());
    EXPECT_TRUE(std::is_sorted(data32.begin(), data32.end()));
}

TEST(AlgoatCTest, BranchlessSearchInt64) {
    std::vector<int64_t> data = {10, 20, 30, 40, 50, 60, 70, 80};
    int64_t idx1 = algoat_search_int64(data.data(), data.size(), 30);
    EXPECT_EQ(idx1, 2);

    int64_t idx2 = algoat_search_int64(data.data(), data.size(), 35);
    EXPECT_EQ(idx2, 3); // lower_bound for 35 is index 3 (value 40)

    int64_t idx3 = algoat_search_int64(data.data(), data.size(), 5);
    EXPECT_EQ(idx3, 0);

    int64_t idx4 = algoat_search_int64(data.data(), data.size(), 100);
    EXPECT_EQ(idx4, 8); // end index
}

TEST(AlgoatCTest, BranchlessSearchFloat64) {
    std::vector<double> data = {-10.5, 0.0, 1.25, 3.5, 100.0};
    int64_t idx1 = algoat_search_float64(data.data(), data.size(), 1.25);
    EXPECT_EQ(idx1, 2);

    int64_t idx2 = algoat_search_float64(data.data(), data.size(), -20.0);
    EXPECT_EQ(idx2, 0);

    int64_t idx3 = algoat_search_float64(data.data(), data.size(), 200.0);
    EXPECT_EQ(idx3, 5);
}

TEST(AlgoatCTest, TopKSelection) {
    std::vector<int64_t> data = {90, 10, 50, 20, 80, 30, 70, 40, 60};
    algoat_topk_int64(data.data(), data.size(), 3);
    // Prefix [0, 3) must be the 3 smallest sorted: 10, 20, 30
    EXPECT_EQ(data[0], 10);
    EXPECT_EQ(data[1], 20);
    EXPECT_EQ(data[2], 30);
}

TEST(AlgoatCTest, EdgeCasesNullAndEmpty) {
    algoat_sort_int64(nullptr, 0, 0);
    algoat_sort_int64(nullptr, 0, 1);
    algoat_sort_int32(nullptr, 0, 0);
    algoat_sort_float64(nullptr, 0, 0);

    int64_t single = 42;
    algoat_sort_int64(&single, 1, 1);
    EXPECT_EQ(single, 42);

    EXPECT_EQ(algoat_search_int64(nullptr, 0, 10), 0);
    EXPECT_EQ(algoat_search_float64(nullptr, 0, 10.0), 0);

    algoat_topk_int64(nullptr, 0, 5);
    algoat_topk_int64(&single, 1, 0);
    algoat_topk_int64(&single, 1, 5);
}
