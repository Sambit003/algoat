#include "algoat/searching/interpolation_search.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace algoat::searching;

class InterpolationSearchTest : public ::testing::Test {
protected:
    InterpolationSearch algo;
};

TEST_F(InterpolationSearchTest, FoundMiddle) {
    std::vector<int> data = {1, 2, 3, 5, 7, 8, 9};
    auto result = algo.search(std::span{data}, 5);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 3);
}

TEST_F(InterpolationSearchTest, NotFound) {
    std::vector<int> data = {1, 2, 3, 5, 7, 8, 9};
    auto result = algo.search(std::span{data}, 42);
    EXPECT_FALSE(result.has_value());
}

TEST_F(InterpolationSearchTest, EmptyInput) {
    std::vector<int> data;
    auto result = algo.search(std::span{data}, 1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(InterpolationSearchTest, FoundFirst) {
    std::vector<int> data = {1, 2, 3, 5, 7, 8, 9};
    auto result = algo.search(std::span{data}, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);
}

TEST_F(InterpolationSearchTest, FoundLast) {
    std::vector<int> data = {1, 2, 3, 5, 7, 8, 9};
    auto result = algo.search(std::span{data}, 9);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 6);
}

TEST_F(InterpolationSearchTest, FoundDuplicates) {
    std::vector<int> data = {1, 2, 3, 3, 3, 8, 9};
    auto result = algo.search(std::span{data}, 3);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value() >= 2 && result.value() <= 4);
}

TEST_F(InterpolationSearchTest, LargeUniformlyDistributed) {
    std::vector<int> data(1000);
    for (int i = 0; i < 1000; ++i) {
        data[i] = i * 2;
    }
    auto result = algo.search(std::span{data}, 400);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 200);
}

TEST_F(InterpolationSearchTest, NonArithmeticFallback) {
    std::vector<std::string> data = {"apple", "banana", "cherry", "date"};
    auto result = algo.search(std::span{data}, std::string("cherry"));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 2);
}

TEST_F(InterpolationSearchTest, PathologicalExtrapolation) {
    std::vector<double> data = {0.0, 10.0, 1000000.0};
    auto result = algo.search(std::span{data}, 500000.0);
    EXPECT_FALSE(result.has_value());
}
