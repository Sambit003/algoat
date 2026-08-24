#include "algoat/sorting/boolean_sort.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <vector>

template <typename T>
void expect_boolean_partition(std::vector<T> data) {
    auto original = data;

    algoat::sorting::sort_boolean(std::span{data});

    // All zero values must come before nonzero values.
    bool found_nonzero = false;

    for (const T value : data) {
        if (value == 0) {
            EXPECT_FALSE(found_nonzero);
        } else {
            found_nonzero = true;
        }
    }

    // The multiset of values must remain unchanged.
    std::sort(original.begin(), original.end());
    std::sort(data.begin(), data.end());

    EXPECT_EQ(data, original);
}

TEST(BooleanSortTest, SortsBinaryValues) {
    std::vector<std::uint8_t> data = {1, 0, 1, 0};

    algoat::sorting::sort_boolean(std::span{data});

    EXPECT_EQ(data, (std::vector<std::uint8_t>{0, 0, 1, 1}));
}

TEST(BooleanSortTest, PreservesArbitraryByteValues) {
    std::vector<std::uint8_t> data = {0, 2, 0, 5, 1, 255};

    expect_boolean_partition(data);
}

TEST(BooleanSortTest, PreservesArbitraryIntegerValues) {
    std::vector<int> data = {5, 0, -3, 2, 0, 100, -7};

    expect_boolean_partition(data);
}

TEST(BooleanSortTest, HandlesEmptyInput) {
    std::vector<int> data;

    EXPECT_NO_THROW(
        algoat::sorting::sort_boolean(std::span{data})
    );

    EXPECT_TRUE(data.empty());
}

TEST(BooleanSortTest, PreservesSizeAndMultiset) {
    std::vector<int> data = {0, 2, 0, 5, -3, 1, 5};
    auto original = data;

    algoat::sorting::sort_boolean(std::span{data});

    // Same number of elements
    EXPECT_EQ(data.size(), original.size());

    // Same values, including duplicates
    std::sort(original.begin(), original.end());
    std::sort(data.begin(), data.end());

    EXPECT_EQ(data, original);
}

