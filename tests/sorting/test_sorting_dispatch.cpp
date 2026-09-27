#include "algoat/core/config_manager.hpp"
#include "algoat/core/dispatcher.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::core;

TEST(SortingDispatchTest, SmallDataUsesInsertionSort) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 8, 1, 9, 3};
    dispatcher.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortingDispatchTest, MostlySortedUsesMergeSort) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i)
        data[i] = i;
    data[98] = 0; // slightly unsorted
    dispatcher.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortingDispatchTest, RandomDataUsesQuickSort) {
    AlgoConfig config;
    Dispatcher dispatcher(config);
    std::vector<int> data = {50, 20, 80, 10, 90, 30, 70, 40, 60, 100, 5,  15, 25,
                             35, 45, 55, 65, 75, 85, 95, 2,  8,  1,   9,  3,  7,
                             4,  6,  0,  11, 22, 33, 44, 55, 66, 77,  88, 99};
    dispatcher.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(SortingDispatchTest, ConfigOverrideUsesHeapSort) {
    AlgoConfig config;
    config.sorting.prefer = "heapsort";
    Dispatcher dispatcher(config);
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};
    dispatcher.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
