#include "algoat/core/dispatcher.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::core;

TEST(DispatcherTest, SortingAutoSelectsInsertionSortForSmall) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 8, 1};
    dispatcher.sort(std::span{data});
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[3], 8);
}

TEST(DispatcherTest, SortingConfigOverride) {
    AlgoConfig config;
    config.sorting.prefer = "heapsort";
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 8, 1};
    dispatcher.sort(std::span{data});
    EXPECT_EQ(data[0], 1);
}

TEST(DispatcherTest, SearchingAutoSelectsLinearForUnsorted) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 8, 1};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 2);
}

TEST(DispatcherTest, SearchingAutoSelectsBinaryForSorted) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {1, 2, 5, 8};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 3);
}

TEST(DispatcherTest, SearchingConfigOverride) {
    AlgoConfig config;
    config.searching.prefer = "linearsearch";
    Dispatcher dispatcher(config);
    std::vector<int> data = {1, 2, 5, 8};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 3);
}

TEST(DispatcherTest, UnknownConfigThrowsWithNoFallback) {
    AlgoConfig config;
    config.sorting.prefer = "unknown";
    config.sorting.fallback = "unknown_fallback";
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 8, 1};
    EXPECT_THROW(dispatcher.sort(std::span{data}), std::runtime_error);
}

TEST(DispatcherTest, SearchingAutoHandlesReverseSorted) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {100, 90, 80, 70, 60, 50, 40, 30, 20, 10};
    auto res = dispatcher.search(std::span{data}, 50);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
}

TEST(DispatcherTest, SearchingAutoHandlesSingleSwappedPair) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    // 10 elements, elements 4 and 5 swapped (normally 40, 50; here 50, 40)
    std::vector<int> data = {10, 20, 30, 50, 40, 60, 70, 80, 90, 100};
    auto res1 = dispatcher.search(std::span{data}, 40);
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), 4);

    auto res2 = dispatcher.search(std::span{data}, 50);
    ASSERT_TRUE(res2.has_value());
    EXPECT_EQ(res2.value(), 3);
}

TEST(DispatcherTest, SearchingAutoHandlesSawtoothDistribution) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {1, 3, 5, 2, 4, 6, 3, 5, 7, 4, 6, 8};
    auto res = dispatcher.search(std::span{data}, 7);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 8);
}

TEST(DispatcherTest, SearchingAutoHandlesLargeSortedDataset) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data(1000000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i * 2);
    }
    auto res = dispatcher.search(std::span{data}, 500000);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 250000);

    auto res_not_found = dispatcher.search(std::span{data}, 500001);
    EXPECT_FALSE(res_not_found.has_value());
}

TEST(DispatcherTest, SearchingAutoNotFoundInUnsorted) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 9, 1, 7, 3};
    auto res = dispatcher.search(std::span{data}, 999);
    EXPECT_FALSE(res.has_value());
}
