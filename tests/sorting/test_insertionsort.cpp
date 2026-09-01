#include "algoat/sorting/insertionsort.hpp"
#include "test_stable_item.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::sorting;
using namespace algoat::sorting::testing;

class InsertionSortTest : public ::testing::Test {
protected:
    InsertionSort algo;
};

TEST_F(InsertionSortTest, IsStable) {
    test_sort_stability(algo);
}

TEST_F(InsertionSortTest, AdaptiveNearlySorted) {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i) {
        data[i] = i;
    }
    std::swap(data[10], data[12]);
    std::swap(data[500], data[505]);
    std::swap(data[990], data[999]);

    algo.sort(std::span{data});

    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
