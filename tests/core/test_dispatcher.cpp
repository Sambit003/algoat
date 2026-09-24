#include "algoat/algoat.hpp"
#include "algoat/core/dispatcher.hpp"

#include <array>
#include <complex>
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

TEST(DispatcherTest, SortingDomainSpecificBool) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::array<bool, 5> data = {true, false, true, false, false};
    dispatcher.sort(std::span<bool>{data});
    std::array<bool, 5> expected = {false, false, false, true, true};
    EXPECT_EQ(data, expected);
}

TEST(DispatcherTest, SortingDomainSpecificComplex) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<std::complex<float>> data = {
        {5.0f, 1.0f}, {1.0f, 0.0f}, {2.0f, 3.0f}, {0.0f, 0.0f}};
    dispatcher.sort(std::span<std::complex<float>>{data});
    std::vector<std::complex<float>> expected = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {5.0f, 1.0f}, {2.0f, 3.0f}};
    EXPECT_EQ(data, expected);
}

TEST(DispatcherTest, PublicAlgoatSortRoutesDomainSpecificTypes) {
    std::array<bool, 4> bool_data = {true, false, true, false};
    algoat::sort(std::span<bool>{bool_data});
    std::array<bool, 4> expected_bool = {false, false, true, true};
    EXPECT_EQ(bool_data, expected_bool);

    std::vector<std::complex<float>> complex_data = {{10.0f, 10.0f}, {0.0f, 0.0f}, {-1.0f, -1.0f}};
    algoat::sort(std::span<std::complex<float>>{complex_data});
    std::vector<std::complex<float>> expected_complex = {
        {-1.0f, -1.0f}, {0.0f, 0.0f}, {10.0f, 10.0f}};
    EXPECT_EQ(complex_data, expected_complex);
}
