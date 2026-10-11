/**
 * @file test_constexpr_sort.cpp
 * @brief Compile-time and runtime verification for constexpr sorting and searching algorithms.
 */

#include "algoat/algoat.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/quicksort.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

// ============================================================================
// Compile-time verification suite (static_assert)
// ============================================================================

// 1. Acceptance Criteria canonical test: algoat::sort on std::array
static_assert([] {
    std::array<int, 6> arr = {9, 2, 7, 1, 5, 4};
    algoat::sort(arr);
    return std::ranges::is_sorted(arr);
}());

// 2. Acceptance Criteria: algoat::sort with custom comparator (descending)
static_assert([] {
    std::array<int, 6> arr = {1, 5, 2, 8, 3, 9};
    algoat::sort(arr, std::ranges::greater{});
    return std::ranges::is_sorted(arr, std::ranges::greater{});
}());

// 3. Acceptance Criteria: algoat::sort with projection
struct Item {
    int key;
    int value;
};

static_assert([] {
    std::array<Item, 4> arr = {{{4, 100}, {1, 200}, {3, 300}, {2, 400}}};
    algoat::sort(arr, std::ranges::less{}, &Item::key);
    return std::ranges::is_sorted(arr, std::ranges::less{}, &Item::key);
}());

// 4. algoat::sorting::introsort iterator & range interface
static_assert([] {
    std::array<int, 7> arr = {15, 3, 2, 10, 8, 4, 1};
    algoat::sorting::introsort(arr.begin(), arr.end());
    return std::ranges::is_sorted(arr);
}());

static_assert([] {
    std::array<int, 5> arr = {5, 4, 3, 2, 1};
    algoat::sorting::introsort(arr);
    return std::ranges::is_sorted(arr);
}());

// 5. algoat::sorting::quicksort iterator & range interface
static_assert([] {
    std::array<int, 8> arr = {12, 5, 3, 8, 5, 1, 10, 2};
    algoat::sorting::quicksort(arr.begin(), arr.end());
    return std::ranges::is_sorted(arr);
}());

static_assert([] {
    std::array<int, 6> arr = {6, 1, 5, 2, 4, 3};
    algoat::sorting::quicksort(arr);
    return std::ranges::is_sorted(arr);
}());

// 6. algoat::sorting::insertion_sort iterator & range interface
static_assert([] {
    std::array<int, 5> arr = {4, 2, 5, 1, 3};
    algoat::sorting::insertion_sort(arr.begin(), arr.end());
    return std::ranges::is_sorted(arr);
}());

static_assert([] {
    std::array<int, 5> arr = {9, 7, 5, 3, 1};
    algoat::sorting::insertionsort(arr);
    return std::ranges::is_sorted(arr);
}());

// 7. algoat::sorting::heapsort iterator & range interface
static_assert([] {
    std::array<int, 7> arr = {10, 20, 15, 30, 5, 25, 0};
    algoat::sorting::heapsort(arr.begin(), arr.end());
    return std::ranges::is_sorted(arr);
}());

static_assert([] {
    std::array<int, 6> arr = {6, 5, 4, 3, 2, 1};
    algoat::sorting::heapsort(arr);
    return std::ranges::is_sorted(arr);
}());

// 8. algoat::searching::binary_search iterator & range interface
static_assert([] {
    std::array<int, 6> arr = {1, 3, 5, 7, 9, 11};
    auto it = algoat::searching::binary_search(arr.begin(), arr.end(), 7);
    return it != arr.end() && *it == 7;
}());

static_assert([] {
    std::array<int, 6> arr = {1, 3, 5, 7, 9, 11};
    auto it = algoat::searching::binary_search(arr.begin(), arr.end(), 8);
    return it == arr.end();
}());

static_assert([] {
    std::array<int, 6> arr = {1, 3, 5, 7, 9, 11};
    auto opt = algoat::searching::binary_search(arr, 7);
    return opt.has_value() && *opt == 3;
}());

static_assert([] {
    std::array<int, 6> arr = {1, 3, 5, 7, 9, 11};
    auto opt = algoat::search(arr, 9);
    return opt.has_value() && *opt == 4;
}());

// 9. C++20 transient allocation with std::vector (P0784R7 & P1004R2)
static_assert([] {
    std::vector<int> vec = {34, 12, 89, 5, 23, 7};
    algoat::sort(vec);
    bool sorted = std::ranges::is_sorted(vec);
    return sorted;
}());

// 10. algoat::sorting::quicksort with custom comparator and projection at compile time
static_assert([] {
    std::array<Item, 4> arr = {{{4, 100}, {1, 200}, {3, 300}, {2, 400}}};
    algoat::sorting::quicksort(arr.begin(), arr.end(), std::ranges::less{}, &Item::key);
    return std::ranges::is_sorted(arr, std::ranges::less{}, &Item::key);
}());

// 11. algoat::searching::binary_search with custom projection and comparator
static_assert([] {
    std::array<Item, 4> arr = {{{1, 100}, {2, 200}, {3, 300}, {4, 400}}};
    auto it = algoat::searching::binary_search(arr.begin(), arr.end(), 3, std::ranges::less{},
                                               &Item::key);
    return it != arr.end() && it->value == 300;
}());

} // namespace

// ============================================================================
// Runtime verification suite (GoogleTest)
// ============================================================================

class ConstexprAlgorithmsTest : public ::testing::Test {
protected:
    std::mt19937 rng{1337};
};

TEST_F(ConstexprAlgorithmsTest, IntrosortBoundaryEdgeCases) {
    // Empty range
    std::vector<int> empty;
    algoat::sorting::introsort(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());

    // Single element
    std::vector<int> single = {42};
    algoat::sorting::introsort(single.begin(), single.end());
    EXPECT_EQ(single.front(), 42);

    // Two elements: sorted and reversed
    std::vector<int> two_sorted = {1, 2};
    algoat::sorting::introsort(two_sorted.begin(), two_sorted.end());
    EXPECT_TRUE(std::ranges::is_sorted(two_sorted));

    std::vector<int> two_rev = {2, 1};
    algoat::sorting::introsort(two_rev.begin(), two_rev.end());
    EXPECT_TRUE(std::ranges::is_sorted(two_rev));

    // Three elements all permutations
    std::vector<int> three = {1, 2, 3};
    do {
        auto copy = three;
        algoat::sorting::introsort(copy.begin(), copy.end());
        EXPECT_TRUE(std::ranges::is_sorted(copy));
    } while (std::next_permutation(three.begin(), three.end()));
}

TEST_F(ConstexprAlgorithmsTest, IntrosortDuplicatesAndLargeSequences) {
    // All duplicates
    std::vector<int> dupes(256, 7);
    algoat::sorting::introsort(dupes.begin(), dupes.end());
    EXPECT_TRUE(std::ranges::is_sorted(dupes));

    // Large random dataset
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), rng);
    algoat::sorting::introsort(data.begin(), data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));
}

TEST_F(ConstexprAlgorithmsTest, QuicksortIteratorAndRangeExecution) {
    std::vector<int> data = {45, 12, 85, 32, 89, 39, 69, 44, 42, 1, 45, 89};
    algoat::sorting::quicksort(data.begin(), data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));

    std::vector<int> data2 = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    algoat::sorting::quicksort(data2);
    EXPECT_TRUE(std::ranges::is_sorted(data2));
}

TEST_F(ConstexprAlgorithmsTest, InsertionSortIteratorAndRangeExecution) {
    std::vector<int> data = {5, 2, 8, 1, 9, 3};
    algoat::sorting::insertion_sort(data.begin(), data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));

    std::vector<int> data2 = {3, 1, 4, 1, 5, 9, 2, 6};
    algoat::sorting::insertionsort(data2, std::ranges::greater{});
    EXPECT_TRUE(std::ranges::is_sorted(data2, std::ranges::greater{}));
}

TEST_F(ConstexprAlgorithmsTest, HeapsortIteratorAndRangeExecution) {
    std::vector<int> data = {19, 48, 99, 12, 33, 44, 21, 5};
    algoat::sorting::heapsort(data.begin(), data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));

    std::vector<int> data2 = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    algoat::sorting::heapsort(data2);
    EXPECT_TRUE(std::ranges::is_sorted(data2));
}

TEST_F(ConstexprAlgorithmsTest, BinarySearchIteratorContract) {
    std::vector<int> sorted_data = {10, 20, 30, 40, 50, 60, 70};

    // Find present elements
    auto it1 = algoat::searching::binary_search(sorted_data.begin(), sorted_data.end(), 30);
    ASSERT_NE(it1, sorted_data.end());
    EXPECT_EQ(*it1, 30);

    auto it2 = algoat::searching::binary_search(sorted_data.begin(), sorted_data.end(), 10);
    ASSERT_NE(it2, sorted_data.end());
    EXPECT_EQ(*it2, 10);

    auto it3 = algoat::searching::binary_search(sorted_data.begin(), sorted_data.end(), 70);
    ASSERT_NE(it3, sorted_data.end());
    EXPECT_EQ(*it3, 70);

    // Missing elements
    auto it_missing = algoat::searching::binary_search(sorted_data.begin(), sorted_data.end(), 25);
    EXPECT_EQ(it_missing, sorted_data.end());

    auto it_below = algoat::searching::binary_search(sorted_data.begin(), sorted_data.end(), 5);
    EXPECT_EQ(it_below, sorted_data.end());

    auto it_above = algoat::searching::binary_search(sorted_data.begin(), sorted_data.end(), 80);
    EXPECT_EQ(it_above, sorted_data.end());
}
