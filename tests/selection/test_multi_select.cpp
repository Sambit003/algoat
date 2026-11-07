#include "algoat/selection/multi_select.hpp"
#include "algoat/selection/selection.hpp"
#include "test_selection_common.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

using algoat::selection::testing::verify_multi_select_invariants;
using algoat::selection::testing::verify_selection_invariant;

} // namespace

TEST(MultiSelectTest, EmptyDataAndEmptyRanks) {
    std::vector<int> empty;
    std::vector<std::size_t> empty_ranks;
    algoat::selection::multi_select(empty, empty_ranks);
    EXPECT_TRUE(empty.empty());

    std::vector<int> data = {42};
    algoat::selection::multi_select(data, empty_ranks);
    EXPECT_EQ(data[0], 42);
}

TEST(MultiSelectTest, SingleElementSingleRank) {
    std::vector<int> single = {99};
    std::vector<std::size_t> ranks = {0};
    algoat::selection::multi_select(single, ranks);
    EXPECT_EQ(single[0], 99);
}

TEST(MultiSelectTest, TwoElementsSingleRank) {
    std::vector<int> data = {42, 17};
    std::vector<std::size_t> ranks = {0};
    algoat::selection::multi_select(data, ranks);
    EXPECT_EQ(data[0], 17);

    data = {42, 17};
    ranks = {1};
    algoat::selection::multi_select(data, ranks);
    EXPECT_EQ(data[1], 42);
}

TEST(MultiSelectTest, TwoElementsBothRanks) {
    std::vector<int> data = {42, 17};
    std::vector<std::size_t> ranks = {0, 1};
    algoat::selection::multi_select(data, ranks);
    EXPECT_EQ(data[0], 17);
    EXPECT_EQ(data[1], 42);
}

TEST(MultiSelectTest, SmallInputQuartiles) {
    std::vector<int> data = {9, 2, 8, 1, 7, 3, 6, 4, 5, 0};
    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::size_t> ranks = {2, 5, 7};
    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, MediumInputDeciles) {
    std::mt19937 gen(42);
    const std::size_t n = 200;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::size_t> ranks = {10, 30, 50, 70, 100, 130, 150, 180, 199};
    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, LargeInputFloydRivestDeciles) {
    std::mt19937 gen(1337);
    const std::size_t n = 5000;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::size_t> ranks = {0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4999};
    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, VeryLargeDatasetVentiles) {
    std::mt19937 gen(999);
    const std::size_t n = 50000;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::size_t> ranks;
    for (std::size_t i = 1; i <= 20; ++i) {
        ranks.push_back((i * n / 20) - 1);
    }

    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, AllElementsIdentical) {
    const std::size_t n = 1500;
    std::vector<int> data(n, 42);
    std::vector<std::size_t> ranks = {0, 10, 100, 750, 1499};

    algoat::selection::multi_select(data, ranks);
    for (int v : data) {
        EXPECT_EQ(v, 42);
    }
}

TEST(MultiSelectTest, HighDuplicateFrequency) {
    std::mt19937 gen(42);
    const std::size_t n = 2000;
    std::vector<int> data(n);
    std::uniform_int_distribution<int> dist(1, 5);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::size_t> ranks = {50, 200, 500, 1000, 1500, 1950};
    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, DuplicateRanks) {
    std::vector<int> data = {19, 3, 15, 7, 2, 8, 12, 1, 9, 5};
    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::size_t> ranks = {2, 2, 5, 5, 5, 8};
    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, DenseRanks) {
    std::mt19937 gen(123);
    const std::size_t n = 100;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    // Selecting 50 ranks out of 100 elements (dense: M = N/2 > N/8)
    std::vector<std::size_t> ranks;
    for (std::size_t i = 0; i < 50; ++i) {
        ranks.push_back(i * 2);
    }

    algoat::selection::multi_select(data, ranks);

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks);
}

TEST(MultiSelectTest, CustomComparatorGreater) {
    std::vector<int> data = {12, 5, 87, 1, 34, 9, 56, 23};
    auto sorted = data;
    std::sort(sorted.begin(), sorted.end(), std::greater<int>{});

    std::vector<std::size_t> ranks = {1, 3, 6};
    algoat::selection::multi_select(data, ranks, std::greater<int>{});

    for (std::size_t r : ranks) {
        EXPECT_EQ(data[r], sorted[r]);
    }
    verify_multi_select_invariants(data, ranks, std::greater<int>{});
}

struct TestRecord {
    int id;
    double score;

    auto operator<=>(const TestRecord& other) const = default;
};

TEST(MultiSelectTest, CustomProjection) {
    std::vector<TestRecord> data = {{1, 9.5}, {2, 1.2}, {3, 5.7}, {4, 3.4},
                                    {5, 8.1}, {6, 2.0}, {7, 7.3}};
    auto sorted = data;
    std::ranges::sort(sorted, {}, &TestRecord::score);

    std::vector<std::size_t> ranks = {1, 3, 5};
    algoat::selection::multi_select(data, ranks, std::ranges::less{}, &TestRecord::score);

    for (std::size_t r : ranks) {
        EXPECT_DOUBLE_EQ(data[r].score, sorted[r].score);
    }
}

TEST(MultiSelectTest, SpanOverload) {
    std::array<int, 6> arr = {50, 10, 40, 20, 60, 30};
    std::array<std::size_t, 2> ranks = {1, 4};

    algoat::selection::multi_select(std::span<int>{arr}, std::span<const std::size_t>{ranks});

    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[4], 50);
}

TEST(QuantilesTest, QuartilesExtraction) {
    const std::size_t n = 100;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::mt19937 gen(42);
    std::shuffle(data.begin(), data.end(), gen);

    auto result = algoat::selection::quantiles(std::span<int>{data}, 4);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 25);
    EXPECT_EQ(result[1], 50);
    EXPECT_EQ(result[2], 75);
}

TEST(QuantilesTest, DecilesExtraction) {
    const std::size_t n = 1000;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::mt19937 gen(101);
    std::shuffle(data.begin(), data.end(), gen);

    auto result = algoat::selection::quantiles(std::span<int>{data}, 10);
    ASSERT_EQ(result.size(), 9u);
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_EQ(result[i], static_cast<int>((i + 1) * 100));
    }
}

TEST(QuantilesTest, EdgeCases) {
    std::vector<int> empty;
    EXPECT_TRUE(algoat::selection::quantiles(std::span<int>{empty}, 4).empty());

    std::vector<int> single = {42};
    EXPECT_TRUE(algoat::selection::quantiles(std::span<int>{single}, 0).empty());
    EXPECT_TRUE(algoat::selection::quantiles(std::span<int>{single}, 1).empty());
    auto single_res = algoat::selection::quantiles(std::span<int>{single}, 2);
    ASSERT_EQ(single_res.size(), 1u);
    EXPECT_EQ(single_res[0], 42);
}

TEST(QuantilesTest, CustomComparatorGreater) {
    const std::size_t n = 100;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::mt19937 gen(77);
    std::shuffle(data.begin(), data.end(), gen);

    auto result = algoat::selection::quantiles(std::span<int>{data}, 4, std::greater<int>{});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 76);
    EXPECT_EQ(result[1], 51);
    EXPECT_EQ(result[2], 26);
}

TEST(MultiSelectTest, ConstexprEvaluation) {
    constexpr auto res = [] {
        std::array<int, 5> a = {5, 2, 4, 1, 3};
        std::array<std::size_t, 2> r = {1, 3};
        algoat::selection::multi_select(a, r);
        return a;
    }();
    static_assert(res[1] == 2);
    static_assert(res[3] == 4);
    EXPECT_EQ(res[1], 2);
    EXPECT_EQ(res[3], 4);
}

TEST(MultiSelectTest, OneMillionTestRunsEquivalenceWithStdSort) {
    // Tests 10^6 total rank selections across varied datasets vs std::sort
    std::mt19937 gen(4242);
    std::size_t total_rank_selections = 0;
    const std::size_t target_selections = 1000000;

    while (total_rank_selections < target_selections) {
        const std::size_t n = 1000;
        std::vector<int> data(n);
        std::uniform_int_distribution<int> val_dist(-50000, 50000);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = val_dist(gen);
        }

        auto sorted = data;
        std::sort(sorted.begin(), sorted.end());

        const std::size_t m = 100;
        std::vector<std::size_t> ranks;
        ranks.reserve(m);
        std::uniform_int_distribution<std::size_t> rank_dist(0, n - 1);
        for (std::size_t i = 0; i < m; ++i) {
            ranks.push_back(rank_dist(gen));
        }
        std::sort(ranks.begin(), ranks.end());

        algoat::selection::multi_select(data, ranks);

        for (std::size_t r : ranks) {
            ASSERT_EQ(data[r], sorted[r]);
        }
        total_rank_selections += m;
    }
    EXPECT_GE(total_rank_selections, target_selections);
}
