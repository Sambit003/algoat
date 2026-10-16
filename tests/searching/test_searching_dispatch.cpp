#include "algoat/core/config_manager.hpp"
#include "algoat/core/dispatcher.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::core;

TEST(SearchingDispatchTest, SortedDataUsesBinarySearch) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {1, 2, 3, 5, 8, 13, 21};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 4);
}

TEST(SearchingDispatchTest, UnsortedDataUsesLinearSearch) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {13, 2, 8, 1, 21, 5, 3};
    auto res = dispatcher.search(std::span{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 2);
}

TEST(SearchingDispatchTest, ConfigOverrideUsesInterpolationSearch) {
    AlgoConfig config;
    config.searching.prefer = "interpolationsearch";
    Dispatcher dispatcher(config);
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80};
    auto res = dispatcher.search(std::span{data}, 60);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
}

TEST(SearchingDispatchTest, ConfigOverrideUsesBranchlessBinarySearch) {
    AlgoConfig config;
    config.searching.prefer = "branchlessbinarysearch";
    Dispatcher dispatcher(config);
    std::vector<int> data = {2, 4, 6, 8, 10, 12, 14, 16};
    auto res = dispatcher.search(std::span{data}, 10);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 4);

    auto res_missing = dispatcher.search(std::span{data}, 7);
    EXPECT_FALSE(res_missing.has_value());
}

TEST(SearchingDispatchTest, ConfigOverrideUsesGallopingSearch) {
    AlgoConfig config;
    config.searching.prefer = "gallopingsearch";
    Dispatcher dispatcher(config);
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15};
    auto res = dispatcher.search(std::span{data}, 7);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 3);
}

TEST(SearchingDispatchTest, SearchHintNearStartRoutesToGallopingSearch) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data(1000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i * 2);
    }
    // Target near start
    auto res = dispatcher.search(std::span{data}, 6, SearchHint::NearStart);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 3);

    auto res_range =
        dispatcher.search(data, 6, std::ranges::less{}, std::identity{}, SearchHint::NearStart);
    ASSERT_TRUE(res_range.has_value());
    EXPECT_EQ(res_range.value(), 3);
}

TEST(SearchingDispatchTest, AutoRoutingNearStartCrossover) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data(1000); // sqrt(1000) ~ 31
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i * 5);
    }
    // Element within first sqrt(N) elements: index 10 (target 50)
    auto res = dispatcher.search(std::span{data}, 50);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 10);

    // Missing element within first sqrt(N) range
    auto res_missing = dispatcher.search(std::span{data}, 49);
    EXPECT_FALSE(res_missing.has_value());
}
