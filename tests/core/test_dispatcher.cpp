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
