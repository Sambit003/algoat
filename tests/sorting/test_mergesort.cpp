#include "algoat/sorting/mergesort.hpp"
#include "test_stable_item.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::sorting;
using namespace algoat::sorting::testing;

class MergeSortTest : public ::testing::Test {
protected:
    MergeSort algo;
};

TEST_F(MergeSortTest, IsStable) {
    test_sort_stability(algo);
}
