#include "algoat/searching/linear_search.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::searching;

class LinearSearchTest : public ::testing::Test {
protected:
    LinearSearch algo;
};

TEST_F(LinearSearchTest, FoundMiddle) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    auto result = algo.search(std::span{data}, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 3);
}

TEST_F(LinearSearchTest, NotFound) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    auto result = algo.search(std::span{data}, 42);
    EXPECT_FALSE(result.has_value());
}

TEST_F(LinearSearchTest, EmptyInput) {
    std::vector<int> data;
    auto result = algo.search(std::span{data}, 1);
    EXPECT_FALSE(result.has_value());
}

TEST_F(LinearSearchTest, FoundFirst) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    auto result = algo.search(std::span{data}, 5);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);
}

TEST_F(LinearSearchTest, FoundLast) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    auto result = algo.search(std::span{data}, 7);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 6);
}

TEST_F(LinearSearchTest, DuplicatesReturnsFirst) {
    std::vector<int> data = {5, 3, 8, 3, 9, 3, 7};
    auto result = algo.search(std::span{data}, 3);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
}

TEST(LinearSearchFreeFunctionTest, FreeFunctionAndPredicate) {
    std::vector<int> data = {5, 3, 8, 1, 9, 2, 7};
    auto result = linear_search(std::span{data}, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3);

    auto result_not_found = linear_search(std::span{data}, 99);
    EXPECT_FALSE(result_not_found.has_value());

    // Custom predicate
    auto result_pred = linear_search(std::span{data}, 8, [](int a, int b) { return a == b; });
    ASSERT_TRUE(result_pred.has_value());
    EXPECT_EQ(*result_pred, 2);
}
