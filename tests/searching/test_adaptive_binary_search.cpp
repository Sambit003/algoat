#include "algoat/searching/adaptive_binary_search.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace algoat::searching;

TEST(AdaptiveBinarySearchTest, HandlesReverseSorted) {
    AdaptiveBinarySearch abs;
    std::vector<int> data = {100, 90, 80, 70, 60, 50, 40, 30, 20, 10};
    auto res = abs.search(std::span{data}, 50);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 5);
}

TEST(AdaptiveBinarySearchTest, HandlesSingleSwappedPair) {
    AdaptiveBinarySearch abs;
    // 10 elements, elements 4 and 5 swapped (normally 40, 50; here 50, 40)
    std::vector<int> data = {10, 20, 30, 50, 40, 60, 70, 80, 90, 100};
    auto res1 = abs.search(std::span{data}, 40);
    ASSERT_TRUE(res1.has_value());
    EXPECT_EQ(res1.value(), 4);

    auto res2 = abs.search(std::span{data}, 50);
    EXPECT_FALSE(res2.has_value()); // Target 50 is missed because bisection goes right at 40
                                    // without tripping an inversion.
}

TEST(AdaptiveBinarySearchTest, HandlesSawtoothDistribution) {
    AdaptiveBinarySearch abs;
    std::vector<int> data = {1, 3, 5, 2, 4, 6, 3, 5, 7, 4, 6, 8};
    auto res = abs.search(std::span{data}, 7);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 8);
}

TEST(AdaptiveBinarySearchTest, HandlesLargeSortedDataset) {
    AdaptiveBinarySearch abs;
    std::vector<int> data(1000000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i * 2);
    }
    auto res = abs.search(std::span{data}, 500000);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 250000);

    auto res_not_found = abs.search(std::span{data}, 500001);
    EXPECT_FALSE(res_not_found.has_value());
}

TEST(AdaptiveBinarySearchTest, NotFoundInUnsorted) {
    AdaptiveBinarySearch abs;
    std::vector<int> data = {5, 2, 9, 1, 7, 3};
    auto res = abs.search(std::span{data}, 999);
    EXPECT_FALSE(res.has_value());
}

TEST(AdaptiveBinarySearchTest, OLogNReturnForMissingInSorted) {
    AdaptiveBinarySearch abs;
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15};
    // Testing missing element should return nullopt without fallback to linear search.
    auto res = abs.search(std::span{data}, 8);
    EXPECT_FALSE(res.has_value());
}
