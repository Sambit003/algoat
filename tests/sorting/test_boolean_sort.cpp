#include "algoat/sorting/boolean_sort.hpp"

#include <gtest/gtest.h>
#include <map>
#include <vector>

using namespace algoat::sorting;

template <typename T> static void verify_boolean_sort_properties(std::vector<T> data) {
    std::map<T, size_t> counts_before;
    for (T x : data) {
        counts_before[x]++;
    }

    sort_boolean(std::span<T>(data));

    // 1. Multiset Conservation Invariant: exact frequency of every element is preserved
    std::map<T, size_t> counts_after;
    for (T x : data) {
        counts_after[x]++;
    }
    EXPECT_EQ(counts_before, counts_after);

    // 2. Partition Invariant: all zeros must precede all non-zeros
    bool seen_non_zero = false;
    for (T x : data) {
        if (x != 0) {
            seen_non_zero = true;
        } else {
            EXPECT_FALSE(seen_non_zero) << "Zero found after non-zero element!";
        }
    }
}

TEST(BooleanSortTest, BoolSpan) {
    bool arr[] = {true, false, true, false, false};
    sort_boolean(std::span<bool>(arr));

    EXPECT_EQ(arr[0], false);
    EXPECT_EQ(arr[1], false);
    EXPECT_EQ(arr[2], false);
    EXPECT_EQ(arr[3], true);
    EXPECT_EQ(arr[4], true);
}

TEST(BooleanSortTest, EmptySpan) {
    std::vector<uint8_t> empty_vec;
    sort_boolean(std::span<uint8_t>(empty_vec));
    EXPECT_TRUE(empty_vec.empty());

    bool empty_bool_arr[] = {false};
    sort_boolean(std::span<bool>(empty_bool_arr, 0));
}

TEST(BooleanSortTest, SingleElement) {
    verify_boolean_sort_properties<int>({42});
    verify_boolean_sort_properties<int>({0});
}

TEST(BooleanSortTest, AllZerosAndAllNonZeros) {
    verify_boolean_sort_properties<int>({0, 0, 0});
    verify_boolean_sort_properties<int>({5, 2, 9});
}

TEST(BooleanSortTest, IntegerMultisetConservation) {
    verify_boolean_sort_properties<uint8_t>({0, 2, 5, 0, 7, 0, 3});
    verify_boolean_sort_properties<int>({-5, 0, 10, 0, -2, 0, 8, 3});
    verify_boolean_sort_properties<int64_t>({0, 1000000, 0, -999999, 0});
}
