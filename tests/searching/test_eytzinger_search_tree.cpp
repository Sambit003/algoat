#include "algoat/searching/eytzinger_search_tree.hpp"

#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

using namespace algoat::searching;

TEST(EytzingerSearchTreeTest, EmptySpan) {
    std::vector<int> data;
    EytzingerSearchTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), 0);
    EXPECT_EQ(tree.lower_bound(10), 0);
}

TEST(EytzingerSearchTreeTest, SingleElementSpan) {
    std::vector<int> data = {42};
    EytzingerSearchTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), 1);

    EXPECT_EQ(tree.lower_bound(10), 0); // target < element -> index 0
    EXPECT_EQ(tree.lower_bound(42), 0); // target == element -> index 0
    EXPECT_EQ(tree.lower_bound(50), 1); // target > element -> index size()
}

TEST(EytzingerSearchTreeTest, SmallArraySearch) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70};
    EytzingerSearchTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), 7);

    // Test exact hits
    for (std::size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(tree.lower_bound(data[i]), i);
    }

    // Test values between elements
    EXPECT_EQ(tree.lower_bound(5), 0);  // < 10 -> 0
    EXPECT_EQ(tree.lower_bound(15), 1); // 10 < 15 <= 20 -> 1
    EXPECT_EQ(tree.lower_bound(25), 2); // 20 < 25 <= 30 -> 2
    EXPECT_EQ(tree.lower_bound(35), 3); // 30 < 35 <= 40 -> 3
    EXPECT_EQ(tree.lower_bound(65), 6); // 60 < 65 <= 70 -> 6
    EXPECT_EQ(tree.lower_bound(75), 7); // > 70 -> size()
}

TEST(EytzingerSearchTreeTest, LargeArrayMatchesStdLowerBound) {
    constexpr std::size_t N = 1000;
    std::vector<int> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int>(i * 2); // 0, 2, 4, ..., 1998
    }

    EytzingerSearchTree<int> tree(std::span<const int>{data});
    EXPECT_EQ(tree.size(), N);

    for (int query = -5; query <= 2005; ++query) {
        auto std_it = std::lower_bound(data.begin(), data.end(), query);
        std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());
        EXPECT_EQ(tree.lower_bound(query), expected_idx) << "Mismatch for query = " << query;
    }
}

TEST(EytzingerSearchTreeTest, DuplicatesHandling) {
    std::vector<int> data = {10, 20, 20, 20, 30, 40};
    EytzingerSearchTree<int> tree(std::span<const int>{data});

    EXPECT_EQ(tree.lower_bound(20), 1); // First index with value >= 20
}

TEST(EytzingerSearchTreeTest, ArrayAccessOperator) {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70};
    EytzingerSearchTree<int> tree(std::span<const int>{data});

    // In-order traversal: node 1 is root (40)
    EXPECT_EQ(tree[1], 40);
}

TEST(EytzingerSearchTreeTest, SupportsGenericTypes) {
    std::vector<std::string> data = {"apple", "banana", "cherry", "date"};
    EytzingerSearchTree<std::string> tree(std::span<const std::string>{data});

    EXPECT_EQ(tree.lower_bound("banana"), 1);
    EXPECT_EQ(tree.lower_bound("coconut"), 3);
    EXPECT_EQ(tree.lower_bound("elderberry"), 4);
}
