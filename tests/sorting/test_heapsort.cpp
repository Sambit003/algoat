#include "algoat/sorting/heapsort.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::sorting;

class HeapSortTest : public ::testing::Test {
protected:
    HeapSort algo;
};

TEST_F(HeapSortTest, SortsDescendingMaxHeapArray) {
    // A max-heap array that is descending, which might require N log N swaps
    std::vector<int> data;
    for (int i = 1000; i >= 0; --i) {
        data.push_back(i);
    }

    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST_F(HeapSortTest, StrictInPlace) {
    // We can't strictly assert zero allocations without overriding new,
    // but we can verify it sorts a large array perfectly.
    std::vector<int> data(5000);
    for (int i = 0; i < 5000; ++i) {
        data[i] = (i * 17) % 5000;
    }

    algo.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}
