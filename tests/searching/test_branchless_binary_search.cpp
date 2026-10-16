#include "algoat/searching/branchless_binary_search.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

using namespace algoat::searching;

TEST(BranchlessBinarySearchTest, EmptyRangeReturnsLast) {
    std::vector<int> data;
    auto it = branchless_lower_bound(data.begin(), data.end(), 42);
    EXPECT_EQ(it, data.end());

    auto range_it = branchless_lower_bound(data, 42);
    EXPECT_EQ(range_it, data.end());
}

TEST(BranchlessBinarySearchTest, SingleElementComparisons) {
    std::vector<int> data = {10};

    // Target smaller
    auto it1 = branchless_lower_bound(data.begin(), data.end(), 5);
    EXPECT_EQ(it1, data.begin());
    EXPECT_EQ(*it1, 10);

    // Target equal
    auto it2 = branchless_lower_bound(data.begin(), data.end(), 10);
    EXPECT_EQ(it2, data.begin());
    EXPECT_EQ(*it2, 10);

    // Target larger
    auto it3 = branchless_lower_bound(data.begin(), data.end(), 15);
    EXPECT_EQ(it3, data.end());
}

TEST(BranchlessBinarySearchTest, TwoElementsComparisons) {
    std::vector<int> data = {10, 20};

    EXPECT_EQ(branchless_lower_bound(data.begin(), data.end(), 5), data.begin());
    EXPECT_EQ(branchless_lower_bound(data.begin(), data.end(), 10), data.begin());
    EXPECT_EQ(branchless_lower_bound(data.begin(), data.end(), 15), data.begin() + 1);
    EXPECT_EQ(branchless_lower_bound(data.begin(), data.end(), 20), data.begin() + 1);
    EXPECT_EQ(branchless_lower_bound(data.begin(), data.end(), 25), data.end());
}

TEST(BranchlessBinarySearchTest, MatchesStdLowerBoundAcrossVaryingSizes) {
    for (std::size_t size : {3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 255, 256, 500, 1024}) {
        std::vector<int> data(size);
        for (std::size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int>(i * 2);
        }

        // Test existing elements, elements in between, and outside bounds
        for (int query = -2; query <= static_cast<int>(size * 2) + 2; ++query) {
            auto expected = std::lower_bound(data.begin(), data.end(), query);
            auto actual = branchless_lower_bound(data.begin(), data.end(), query);
            ASSERT_EQ(actual, expected) << "Failed for size=" << size << " and query=" << query;

            auto range_actual = branchless_lower_bound(data, query);
            ASSERT_EQ(range_actual, expected)
                << "Range overload failed for size=" << size << " and query=" << query;
        }
    }
}

TEST(BranchlessBinarySearchTest, HandlesDuplicatesReturningFirstOccurrence) {
    std::vector<int> data = {1, 2, 4, 4, 4, 4, 7, 9};
    auto it = branchless_lower_bound(data.begin(), data.end(), 4);
    EXPECT_EQ(it, data.begin() + 2);
    EXPECT_EQ(*it, 4);

    auto std_it = std::lower_bound(data.begin(), data.end(), 4);
    EXPECT_EQ(it, std_it);
}

TEST(BranchlessBinarySearchTest, SupportsCustomComparatorDescending) {
    std::vector<int> data = {90, 70, 50, 30, 10};
    auto comp = std::ranges::greater{};

    auto it = branchless_lower_bound(data.begin(), data.end(), 50, comp);
    EXPECT_EQ(it, data.begin() + 2);
    EXPECT_EQ(*it, 50);

    auto it_missing = branchless_lower_bound(data.begin(), data.end(), 60, comp);
    EXPECT_EQ(it_missing, data.begin() + 2);
    EXPECT_EQ(*it_missing, 50);

    auto it_end = branchless_lower_bound(data.begin(), data.end(), 5, comp);
    EXPECT_EQ(it_end, data.end());
}

struct Item {
    int id;
    std::string name;
};

TEST(BranchlessBinarySearchTest, SupportsCustomProjection) {
    std::vector<Item> items = {{1, "alpha"}, {3, "beta"}, {5, "gamma"}, {7, "delta"}};
    auto it = branchless_lower_bound(items.begin(), items.end(), 5, {}, &Item::id);
    ASSERT_NE(it, items.end());
    EXPECT_EQ(it->id, 5);
    EXPECT_EQ(it->name, "gamma");

    auto it_missing = branchless_lower_bound(items.begin(), items.end(), 4, {}, &Item::id);
    ASSERT_NE(it_missing, items.end());
    EXPECT_EQ(it_missing->id, 5);
}

TEST(BranchlessBinarySearchTest, ConstexprEvaluation) {
    static constexpr std::array<int, 5> data = {2, 4, 6, 8, 10};
    constexpr auto it = branchless_lower_bound(data.begin(), data.end(), 6);
    static_assert(*it == 6);
    static_assert(it == data.begin() + 2);

    constexpr auto it_end = branchless_lower_bound(data.begin(), data.end(), 12);
    static_assert(it_end == data.end());
}

TEST(BranchlessBinarySearchTest, StructConformsToSearchAlgorithmConcept) {
    BranchlessBinarySearch searcher;
    EXPECT_EQ(searcher.name(), "branchlessbinarysearch");
    EXPECT_TRUE(searcher.requires_sorted());

    std::vector<int> data = {1, 3, 5, 7, 9};
    auto res = searcher.search(std::span<const int>{data}, 5);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 2);

    auto res_missing = searcher.search(std::span<const int>{data}, 6);
    EXPECT_FALSE(res_missing.has_value());
}
