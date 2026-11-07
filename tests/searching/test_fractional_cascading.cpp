/**
 * @file test_fractional_cascading.cpp
 * @brief Unit tests for FractionalCascading multi-list search engine.
 */

#include "algoat/searching/fractional_cascading.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <vector>

using namespace algoat::searching;

namespace {

template <typename T, typename Compare = std::ranges::less>
void assert_all_match_std_lower_bound(const FractionalCascading<T, Compare>& fc,
                                      const std::vector<std::vector<T>>& lists, const T& target,
                                      Compare comp = Compare{}) {
    auto results = fc.search_all(target);
    ASSERT_EQ(results.size(), lists.size());

    std::vector<std::size_t> span_buf(lists.size());
    fc.search_all_to_span(target, span_buf);
    EXPECT_EQ(span_buf, results);

    for (std::size_t i = 0; i < lists.size(); ++i) {
        auto expected_it = std::lower_bound(lists[i].begin(), lists[i].end(), target, comp);
        std::size_t expected_idx = static_cast<std::size_t>(expected_it - lists[i].begin());
        EXPECT_EQ(results[i], expected_idx) << "Mismatch on list " << i << " for target " << target;
    }
}

} // namespace

TEST(FractionalCascadingTest, EmptyRangeOfLists) {
    std::vector<std::vector<int>> lists;
    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), 0);
    EXPECT_EQ(fc.total_augmented_elements(), 0);

    auto res = fc.search_all(42);
    EXPECT_TRUE(res.empty());

    std::vector<std::size_t> span_buf;
    fc.search_all_to_span(42, span_buf);
    EXPECT_TRUE(span_buf.empty());
}

TEST(FractionalCascadingTest, CompletelyEmptyLists) {
    std::vector<std::vector<int>> lists(5);
    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), 5);
    EXPECT_EQ(fc.total_augmented_elements(), 0);

    for (int q : {-10, 0, 42, 100}) {
        assert_all_match_std_lower_bound(fc, lists, q);
    }
}

TEST(FractionalCascadingTest, SingleListEquivalence) {
    std::vector<std::vector<int>> lists = {{10, 20, 30, 40, 50}};
    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), 1);

    for (int target : {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}) {
        assert_all_match_std_lower_bound(fc, lists, target);
    }
}

TEST(FractionalCascadingTest, DeBergWorkedExample) {
    // Reference: de Berg et al., Computational Geometry, Section 5.6
    std::vector<std::vector<int>> lists = {{5, 15, 25}, {10, 20, 30}, {3, 8, 12, 18, 28}};
    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), 3);

    // Target = 17:
    // in L_0: smallest >= 17 is 25 (index 2)
    // in L_1: smallest >= 17 is 20 (index 1)
    // in L_2: smallest >= 17 is 18 (index 3)
    auto res = fc.search_all(17);
    ASSERT_EQ(res.size(), 3);
    EXPECT_EQ(res[0], 2);
    EXPECT_EQ(res[1], 1);
    EXPECT_EQ(res[2], 3);

    std::vector<std::size_t> span_res(3, 0);
    fc.search_all_to_span(17, span_res);
    EXPECT_EQ(span_res[0], 2);
    EXPECT_EQ(span_res[1], 1);
    EXPECT_EQ(span_res[2], 3);
}

TEST(FractionalCascadingTest, BoundaryQueriesAllLists) {
    std::vector<std::vector<int>> lists = {{10, 20, 30}, {15, 25, 35}, {5, 40, 50}};
    FractionalCascading<int> fc(lists);

    // Target smaller than all elements
    assert_all_match_std_lower_bound(fc, lists, 1);

    // Target greater than all elements
    assert_all_match_std_lower_bound(fc, lists, 100);
}

TEST(FractionalCascadingTest, EmptyInnerLists) {
    std::vector<std::vector<int>> lists = {{}, {10, 20}, {}, {5, 15, 25}, {}};
    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), 5);

    for (int q : {0, 5, 10, 15, 20, 25, 30}) {
        assert_all_match_std_lower_bound(fc, lists, q);
    }
}

TEST(FractionalCascadingTest, DuplicatesInLists) {
    std::vector<std::vector<int>> lists = {
        {2, 2, 5, 5, 5, 8}, {1, 2, 2, 4, 4, 9}, {2, 3, 3, 3, 5, 7}};
    FractionalCascading<int> fc(lists);

    for (int q = 0; q <= 10; ++q) {
        assert_all_match_std_lower_bound(fc, lists, q);
    }
}

TEST(FractionalCascadingTest, CustomComparatorGreater) {
    std::vector<std::vector<int>> lists = {{50, 40, 30, 20, 10}, {55, 35, 15}, {60, 45, 25, 5}};
    FractionalCascading<int, std::greater<int>> fc(lists, std::greater<int>{});

    for (int q = 0; q <= 65; q += 5) {
        assert_all_match_std_lower_bound(fc, lists, q, std::greater<int>{});
    }
}

TEST(FractionalCascadingTest, MemoryFootprintStrictlyBounded) {
    // Acceptance criteria: Memory footprint strictly bounded by <= 2 sum(|L_i|) *
    // sizeof(AugmentedElement).
    std::vector<std::vector<int>> lists = {{1, 4, 7, 10, 13, 16, 19},
                                           {2, 5, 8, 11, 14, 17, 20, 23},
                                           {3, 6, 9, 12, 15, 18, 21, 24, 27},
                                           {4, 8, 12, 16, 20, 24, 28, 32, 36, 40}};

    std::size_t total_input_elements = 0;
    for (const auto& l : lists) {
        total_input_elements += l.size();
    }

    FractionalCascading<int> fc(lists);
    EXPECT_LE(fc.total_augmented_elements(), 2 * total_input_elements);
}

TEST(FractionalCascadingTest, EquivalenceLargeKLists) {
    constexpr std::size_t K = 50;
    constexpr std::size_t N = 1000;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> val_dist(0, 100000);

    std::vector<std::vector<int>> lists(K);
    for (std::size_t i = 0; i < K; ++i) {
        lists[i].resize(N);
        for (std::size_t j = 0; j < N; ++j) {
            lists[i][j] = val_dist(rng);
        }
        std::sort(lists[i].begin(), lists[i].end());
    }

    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), K);

    std::uniform_int_distribution<int> query_dist(-500, 105000);
    for (int trial = 0; trial < 100; ++trial) {
        int target = query_dist(rng);
        assert_all_match_std_lower_bound(fc, lists, target);
    }
}

TEST(FractionalCascadingTest, EquivalenceK50ListsN100K) {
    // Acceptance criteria: 100% equivalence of search_all results with independent
    // std::lower_bound calls across K = 50 lists with N = 10^5 elements each.
    constexpr std::size_t K = 50;
    constexpr std::size_t N = 100000;
    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<int> val_dist(0, 10000000);

    std::vector<std::vector<int>> lists(K);
    std::size_t total_elements = 0;
    for (std::size_t i = 0; i < K; ++i) {
        lists[i].resize(N);
        for (std::size_t j = 0; j < N; ++j) {
            lists[i][j] = val_dist(rng);
        }
        std::sort(lists[i].begin(), lists[i].end());
        total_elements += N;
    }

    FractionalCascading<int> fc(lists);
    EXPECT_EQ(fc.list_count(), K);
    EXPECT_LE(fc.total_augmented_elements(), 2 * total_elements);

    std::uniform_int_distribution<int> query_dist(-1000, 10001000);
    for (int trial = 0; trial < 50; ++trial) {
        int target = query_dist(rng);
        assert_all_match_std_lower_bound(fc, lists, target);
    }
}
