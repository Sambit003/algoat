#include "algoat/core/traits.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::core;

TEST(DataTraitsTest, EmptyData) {
    std::vector<int> data;
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 0);
    EXPECT_DOUBLE_EQ(traits.sortedness_ratio, 1.0);
    EXPECT_FALSE(traits.has_duplicates);
}

TEST(DataTraitsTest, SingleElement) {
    std::vector<int> data = {42};
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 1);
    EXPECT_DOUBLE_EQ(traits.sortedness_ratio, 1.0);
    EXPECT_FALSE(traits.has_duplicates);
}

TEST(DataTraitsTest, FullySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 5);
    EXPECT_DOUBLE_EQ(traits.sortedness_ratio, 1.0);
    EXPECT_FALSE(traits.has_duplicates);
}

TEST(DataTraitsTest, ReverseSorted) {
    std::vector<int> data = {5, 4, 3, 2, 1};
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 5);
    EXPECT_DOUBLE_EQ(traits.sortedness_ratio, 0.0);
    EXPECT_FALSE(traits.has_duplicates);
}

TEST(DataTraitsTest, RandomData) {
    std::vector<int> data = {3, 1, 4, 2, 5};
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 5);
    // pairs: (3,1)=F, (1,4)=T, (4,2)=F, (2,5)=T. 2 sorted pairs out of 4.
    EXPECT_DOUBLE_EQ(traits.sortedness_ratio, 0.5);
    EXPECT_FALSE(traits.has_duplicates);
}

TEST(DataTraitsTest, AllDuplicates) {
    std::vector<int> data = {7, 7, 7, 7};
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 4);
    // pairs: (7,7)=T, (7,7)=T, (7,7)=T. 3 sorted pairs out of 3.
    EXPECT_DOUBLE_EQ(traits.sortedness_ratio, 1.0);
    EXPECT_TRUE(traits.has_duplicates);
    EXPECT_TRUE(traits.is_exact);
}

TEST(DataTraitsTest, LargeFullySortedSublinear) {
    std::vector<int> data(50000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i);
    }
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 50000);
    EXPECT_GE(traits.sortedness_ratio, 0.99);
    EXPECT_FALSE(traits.is_exact);
}

TEST(DataTraitsTest, LargeReverseSortedSublinear) {
    std::vector<int> data(50000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(data.size() - i);
    }
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 50000);
    EXPECT_LE(traits.sortedness_ratio, 0.05);
    EXPECT_FALSE(traits.is_exact);
}

TEST(DataTraitsTest, LargeMostlySortedSublinear) {
    std::vector<int> data(50000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i);
    }
    // Perturb a few elements
    data[100] = 500;
    data[200] = 100;
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 50000);
    EXPECT_GE(traits.sortedness_ratio, 0.95);
    EXPECT_FALSE(traits.is_exact);
}

TEST(DataTraitsTest, LargeFloatSequenceSublinear) {
    std::vector<float> data(50000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i) * 0.5f;
    }
    auto traits = analyze(data);
    EXPECT_EQ(traits.size, 50000);
    EXPECT_GE(traits.sortedness_ratio, 0.99);
    EXPECT_FALSE(traits.is_exact);
}
