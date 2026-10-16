#include "algoat/searching/galloping_search.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

using namespace algoat::searching;

TEST(GallopingSearchTest, EmptyRangeReturnsFirst) {
    std::vector<int> data;
    auto it = galloping_search(data.begin(), data.end(), 42);
    EXPECT_EQ(it, data.end());

    auto range_it = galloping_search(data, 42);
    EXPECT_EQ(range_it, data.end());
}

TEST(GallopingSearchTest, SingleElementComparisons) {
    std::vector<int> data = {10};

    // Target smaller
    auto it1 = galloping_search(data.begin(), data.end(), 5);
    EXPECT_EQ(it1, data.begin());
    EXPECT_EQ(*it1, 10);

    // Target equal
    auto it2 = galloping_search(data.begin(), data.end(), 10);
    EXPECT_EQ(it2, data.begin());
    EXPECT_EQ(*it2, 10);

    // Target larger
    auto it3 = galloping_search(data.begin(), data.end(), 15);
    EXPECT_EQ(it3, data.end());
}

TEST(GallopingSearchTest, TargetAtStartFastPath) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80};

    // Target <= first element
    EXPECT_EQ(galloping_search(data.begin(), data.end(), 5), data.begin());
    EXPECT_EQ(galloping_search(data.begin(), data.end(), 10), data.begin());
}

TEST(GallopingSearchTest, TargetOnDoublingSteps) {
    std::vector<int> data(128);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i * 10);
    }

    // Indices: 1, 2, 4, 8, 16, 32, 64
    for (std::size_t idx : {1, 2, 4, 8, 16, 32, 64}) {
        int target = data[idx];
        auto it = galloping_search(data.begin(), data.end(), target);
        EXPECT_EQ(it, data.begin() + idx);
        EXPECT_EQ(*it, target);

        // Check target - 1 (in between)
        auto it_prev = galloping_search(data.begin(), data.end(), target - 1);
        EXPECT_EQ(it_prev, data.begin() + idx);
    }
}

TEST(GallopingSearchTest, MatchesStdLowerBoundAcrossVaryingSizes) {
    for (std::size_t size : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 50, 100, 256, 500, 1000}) {
        std::vector<int> data(size);
        for (std::size_t i = 0; i < size; ++i) {
            data[i] = static_cast<int>(i * 3);
        }

        for (int query = -3; query <= static_cast<int>(size * 3) + 3; ++query) {
            auto expected = std::lower_bound(data.begin(), data.end(), query);
            auto actual = galloping_search(data.begin(), data.end(), query);
            ASSERT_EQ(actual, expected)
                << "Galloping search failed for size=" << size << " and query=" << query;

            auto range_actual = galloping_search(data, query);
            ASSERT_EQ(range_actual, expected)
                << "Range galloping search failed for size=" << size << " and query=" << query;
        }
    }
}

TEST(GallopingSearchTest, HandlesDuplicatesReturningFirstOccurrence) {
    std::vector<int> data = {2, 5, 5, 5, 5, 9, 12};
    auto it = galloping_search(data.begin(), data.end(), 5);
    EXPECT_EQ(it, data.begin() + 1);
    EXPECT_EQ(*it, 5);

    auto std_it = std::lower_bound(data.begin(), data.end(), 5);
    EXPECT_EQ(it, std_it);
}

TEST(GallopingSearchTest, SupportsCustomComparatorDescending) {
    std::vector<int> data = {100, 80, 60, 40, 20, 0};
    auto comp = std::ranges::greater{};

    auto it = galloping_search(data.begin(), data.end(), 60, comp);
    EXPECT_EQ(it, data.begin() + 2);
    EXPECT_EQ(*it, 60);

    auto it_between = galloping_search(data.begin(), data.end(), 70, comp);
    EXPECT_EQ(it_between, data.begin() + 2);
    EXPECT_EQ(*it_between, 60);

    auto it_head = galloping_search(data.begin(), data.end(), 120, comp);
    EXPECT_EQ(it_head, data.begin());

    auto it_end = galloping_search(data.begin(), data.end(), -10, comp);
    EXPECT_EQ(it_end, data.end());
}

struct Record {
    int key;
    std::string value;
};

TEST(GallopingSearchTest, SupportsCustomProjection) {
    std::vector<Record> records = {{10, "ten"}, {20, "twenty"}, {30, "thirty"}, {40, "forty"}};
    auto it = galloping_search(records.begin(), records.end(), 30, {}, &Record::key);
    ASSERT_NE(it, records.end());
    EXPECT_EQ(it->key, 30);
    EXPECT_EQ(it->value, "thirty");

    auto it_missing = galloping_search(records.begin(), records.end(), 25, {}, &Record::key);
    ASSERT_NE(it_missing, records.end());
    EXPECT_EQ(it_missing->key, 30);
}

TEST(GallopingSearchTest, ConstexprEvaluation) {
    static constexpr std::array<int, 6> data = {1, 3, 5, 7, 9, 11};
    constexpr auto it = galloping_search(data.begin(), data.end(), 7);
    static_assert(*it == 7);
    static_assert(it == data.begin() + 3);

    constexpr auto it_start = galloping_search(data.begin(), data.end(), 0);
    static_assert(it_start == data.begin());

    constexpr auto it_end = galloping_search(data.begin(), data.end(), 20);
    static_assert(it_end == data.end());
}

TEST(GallopingSearchTest, StructConformsToSearchAlgorithmConcept) {
    GallopingSearch searcher;
    EXPECT_EQ(searcher.name(), "gallopingsearch");
    EXPECT_TRUE(searcher.requires_sorted());

    std::vector<int> data = {2, 4, 6, 8, 10};
    auto res = searcher.search(std::span<const int>{data}, 8);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 3);

    auto res_missing = searcher.search(std::span<const int>{data}, 7);
    EXPECT_FALSE(res_missing.has_value());
}
