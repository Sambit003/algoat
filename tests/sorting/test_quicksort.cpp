#include "algoat/sorting/quicksort.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::sorting;

class QuickSortTest : public ::testing::Test {
protected:
    QuickSort algo;

    void verify_sort(std::vector<int>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

TEST_F(QuickSortTest, RandomData) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    verify_sort(data);
}

TEST_F(QuickSortTest, EmptyInput) {
    std::vector<int> data;
    verify_sort(data);
}

TEST_F(QuickSortTest, SingleElement) {
    std::vector<int> data = {42};
    verify_sort(data);
}

TEST_F(QuickSortTest, AllDuplicates) {
    std::vector<int> data = {7, 7, 7, 7, 7};
    verify_sort(data);
}

TEST_F(QuickSortTest, AlreadySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7};
    verify_sort(data);
}

TEST_F(QuickSortTest, ReverseSorted) {
    std::vector<int> data = {7, 6, 5, 4, 3, 2, 1};
    verify_sort(data);
}

TEST_F(QuickSortTest, LargeRandom) {
    std::vector<int> data(1000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 10000);
    for (int& x : data) {
        x = dist(gen);
    }
    verify_sort(data);
}

TEST_F(QuickSortTest, LowEntropyAllSame) {
    std::vector<int> data(10000, 42);
    verify_sort(data);
}

TEST_F(QuickSortTest, LowEntropyFewDistinct) {
    std::vector<int> data;
    for (int i = 0; i < 5000; ++i) {
        data.push_back(i % 3);
    }
    std::mt19937 gen(99);
    std::shuffle(data.begin(), data.end(), gen);
    verify_sort(data);
}
