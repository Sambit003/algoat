#include "algoat/core/config_manager.hpp"
#include "algoat/core/dispatcher.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::core;

TEST(SearchingDispatchTest, SortedDataUsesBinarySearch) {
    AlgoConfig config;
    ConfigManager::instance().update_config(config);
    Dispatcher dispatcher;
    std::vector<int> data = {1, 2, 3, 5, 8, 13, 21};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 4);
}

TEST(SearchingDispatchTest, UnsortedDataUsesLinearSearch) {
    AlgoConfig config;
    ConfigManager::instance().update_config(config);
    Dispatcher dispatcher;
    std::vector<int> data = {13, 2, 8, 1, 21, 5, 3};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 2);
}

TEST(SearchingDispatchTest, ConfigOverrideUsesInterpolationSearch) {
    AlgoConfig config;
    config.searching.prefer = "interpolationsearch";
    ConfigManager::instance().update_config(config);
    Dispatcher dispatcher;
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80};
    auto res = dispatcher.search(std::span{data}, 60);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
}
