#include "algoat/sorting/quicksort.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::sorting;

class QuickSortTest : public ::testing::Test {
protected:
    QuickSort algo;
};

TEST_F(QuickSortTest, HandlesPipeOrganData) {
    int n = 1000;
    std::vector<int> data(n);
    for (int i = 0; i < n / 2; ++i) {
        data[i] = i;
        data[n - 1 - i] = i;
    }

    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST_F(QuickSortTest, HandlesDutchFlagDuplicates) {
    // Mostly three values
    std::vector<int> data;
    for (int i = 0; i < 500; ++i) {
        data.push_back(0);
        data.push_back(1);
        data.push_back(2);
    }
    std::random_device rd;
    std::mt19937 g(42);
    std::shuffle(data.begin(), data.end(), g);

    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
