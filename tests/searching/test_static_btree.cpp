#include "algoat/searching/static_btree.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace algoat::searching;

TEST(StaticBTreeTest, EmptySpan) {
    std::vector<int> data;
    StaticBTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), 0);
    EXPECT_EQ(tree.lower_bound(10), 0);

    std::vector<int> queries = {1, 2, 3};
    std::vector<std::size_t> results(3, 999);
    tree.lower_bound_batch(std::span<const int>{queries}, std::span<std::size_t>{results});
    for (std::size_t res : results) {
        EXPECT_EQ(res, 0);
    }
}

TEST(StaticBTreeTest, SingleElementSpan) {
    std::vector<int> data = {42};
    StaticBTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), 1);

    EXPECT_EQ(tree.lower_bound(10), 0); // target < element -> index 0
    EXPECT_EQ(tree.lower_bound(42), 0); // target == element -> index 0
    EXPECT_EQ(tree.lower_bound(50), 1); // target > element -> index size()
}

TEST(StaticBTreeTest, SmallArraySearch) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70};
    StaticBTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), 7);

    // Exact hits
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(tree.lower_bound(data[i]), i) << "Mismatch at index " << i;
    }

    // Interleaved query values
    EXPECT_EQ(tree.lower_bound(5), 0);
    EXPECT_EQ(tree.lower_bound(15), 1);
    EXPECT_EQ(tree.lower_bound(25), 2);
    EXPECT_EQ(tree.lower_bound(35), 3);
    EXPECT_EQ(tree.lower_bound(65), 6);
    EXPECT_EQ(tree.lower_bound(75), 7);
}

TEST(StaticBTreeTest, ExactlyFifteenAndSixteenElements) {
    // Tests boundary around B = 15 keys
    for (std::size_t n : {15, 16, 30, 31, 32}) {
        std::vector<int> data(n);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = static_cast<int>(i * 10);
        }
        StaticBTree<int> tree(std::span<const int>{data});
        EXPECT_EQ(tree.size(), n);

        for (int q = -5; q <= static_cast<int>(n * 10 + 5); ++q) {
            auto it = std::lower_bound(data.begin(), data.end(), q);
            std::size_t expected = static_cast<std::size_t>(it - data.begin());
            EXPECT_EQ(tree.lower_bound(q), expected) << "Failed for n = " << n << ", q = " << q;
        }
    }
}

TEST(StaticBTreeTest, LargeArrayMatchesStdLowerBound) {
    constexpr std::size_t N = 2000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i * 3);
    }

    StaticBTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), N);

    for (int query = -10; query <= static_cast<int>(N * 3 + 10); ++query) {
        auto std_it = std::lower_bound(data.begin(), data.end(), query);
        std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());
        EXPECT_EQ(tree.lower_bound(query), expected_idx) << "Mismatch for query = " << query;
    }
}

TEST(StaticBTreeTest, DuplicatesHandling) {
    std::vector<int> data = {10, 20, 20, 20, 30, 40};
    StaticBTree<int> tree(std::span<const int>{data});

    EXPECT_EQ(tree.lower_bound(20), 1); // First index with value >= 20
    EXPECT_EQ(tree.lower_bound(10), 0);
    EXPECT_EQ(tree.lower_bound(30), 4);

    // Large-scale tests with duplicate values: N = 10^2, 10^3, 10^4, 10^5
    for (std::size_t n : {100, 1000, 10000, 100000}) {
        std::vector<int> dup_data(n);
        for (std::size_t i = 0; i < n; ++i) {
            dup_data[i] = static_cast<int>(i / 7); // Generates groups of 7 identical elements
        }

        StaticBTree<int> dup_tree(std::span<const int>{dup_data});
        EXPECT_EQ(dup_tree.size(), n);

        const int max_val = dup_data.back();
        for (int step = 0; step < 200; ++step) {
            int q = (step * (max_val + 10)) / 200 - 5;
            auto std_it = std::lower_bound(dup_data.begin(), dup_data.end(), q);
            std::size_t expected_idx = static_cast<std::size_t>(std_it - dup_data.begin());
            EXPECT_EQ(dup_tree.lower_bound(q), expected_idx)
                << "Duplicate mismatch at n=" << n << " query=" << q;
        }
    }
}

TEST(StaticBTreeTest, BatchedQueryMatchesSingleQuery) {
    constexpr std::size_t N = 5000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i * 2);
    }

    StaticBTree<int> tree(std::span<const int>{data});

    std::vector<int> queries;
    for (int q = -10; q < 500; ++q) {
        queries.push_back(q);
    }

    std::vector<std::size_t> batched_results(queries.size());
    tree.lower_bound_batch(std::span<const int>{queries}, std::span<std::size_t>{batched_results});

    for (std::size_t i = 0; i < queries.size(); ++i) {
        std::size_t expected = tree.lower_bound(queries[i]);
        EXPECT_EQ(batched_results[i], expected)
            << "Batch mismatch at query index " << i << ", query = " << queries[i];
    }
}

TEST(StaticBTreeTest, GenericTypes) {
    std::vector<std::string> str_data = {"alpha", "beta", "delta", "gamma", "omega"};
    StaticBTree<std::string> str_tree(std::span<const std::string>{str_data});

    EXPECT_EQ(str_tree.lower_bound("beta"), 1);
    EXPECT_EQ(str_tree.lower_bound("charlie"), 2);
    EXPECT_EQ(str_tree.lower_bound("zebra"), 5);

    std::vector<double> dbl_data = {1.1, 2.2, 3.3, 4.4, 5.5};
    StaticBTree<double> dbl_tree(std::span<const double>{dbl_data});

    EXPECT_EQ(dbl_tree.lower_bound(2.2), 1);
    EXPECT_EQ(dbl_tree.lower_bound(3.0), 2);
    EXPECT_EQ(dbl_tree.lower_bound(6.0), 5);
}
