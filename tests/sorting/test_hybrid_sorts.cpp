#include "algoat/sorting/blocksort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/timsort.hpp"
#include "test_stable_item.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::sorting;
using namespace algoat::sorting::testing;

// --- TimSort Tests ---
class TimSortTest : public ::testing::Test {
protected:
    TimSort algo;
};

TEST_F(TimSortTest, IsStable) {
    test_sort_stability(algo);
}

TEST_F(TimSortTest, SawtoothRuns) {
    std::vector<int> data;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 20; ++j) {
            data.push_back(j);
        }
        for (int j = 20; j > 0; --j) {
            data.push_back(j);
        }
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

// --- BlockSort Tests ---
class BlockSortTest : public ::testing::Test {
protected:
    BlockSort algo;
};

TEST_F(BlockSortTest, IsStable) {
    test_sort_stability(algo);
}

// --- IntroSort Tests ---
class IntroSortTest : public ::testing::Test {
protected:
    IntroSort algo;
};

TEST_F(IntroSortTest, HandlesMedianOfThreeKillerSequence) {
    int n = 10000;
    std::vector<int> data(n);
    for (int i = 0; i < n / 2; ++i) {
        data[i] = i * 2;
        data[n - 1 - i] = i * 2 + 1;
    }
    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
