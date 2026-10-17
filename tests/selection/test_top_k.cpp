#include "algoat/selection/selection.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

namespace {

struct Student {
    std::string name;
    int rank;

    auto operator<=>(const Student& other) const = default;
};

} // namespace

TEST(TopKTest, UnsortedEmptyAndZero) {
    std::vector<int> empty;
    algoat::selection::top_k_unsorted(empty, 0);
    algoat::selection::top_k_unsorted(empty, 5);
    EXPECT_TRUE(empty.empty());

    std::vector<int> data = {5, 4, 3, 2, 1};
    algoat::selection::top_k_unsorted(data, 0);
    // Unchanged
    EXPECT_EQ(data.size(), 5UL);
}

TEST(TopKTest, SortedEmptyAndZero) {
    std::vector<int> empty;
    algoat::selection::top_k_sorted(empty, 0);
    algoat::selection::top_k_sorted(empty, 5);
    EXPECT_TRUE(empty.empty());

    std::vector<int> data = {5, 4, 3, 2, 1};
    algoat::selection::top_k_sorted(data, 0);
    EXPECT_EQ(data.size(), 5UL);
}

TEST(TopKTest, UnsortedKGreaterThanSize) {
    std::vector<int> data = {9, 3, 1, 7};
    algoat::selection::top_k_unsorted(data, 10);
    EXPECT_EQ(data.size(), 4UL);
}

TEST(TopKTest, SortedKGreaterThanSize) {
    std::vector<int> data = {9, 3, 1, 7};
    algoat::selection::top_k_sorted(data, 10);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(TopKTest, TopKUnsortedInvariants) {
    std::mt19937 gen(42);
    const std::size_t n = 200;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t k : {1UL, 5UL, 50UL, 100UL, 150UL, n - 1}) {
        auto copy = data;
        algoat::selection::top_k_unsorted(copy, k);

        // All elements in [0, k) must be <= all elements in [k, n)
        auto max_in_prefix = *std::max_element(copy.begin(), copy.begin() + k);
        auto min_in_suffix = *std::min_element(copy.begin() + k, copy.end());
        EXPECT_LE(max_in_prefix, min_in_suffix);

        // The set of elements in [0, k) must match the k smallest of data
        auto sorted_copy = copy;
        std::sort(sorted_copy.begin(), sorted_copy.begin() + k);
        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_EQ(sorted_copy[i], static_cast<int>(i + 1));
        }
    }
}

TEST(TopKTest, TopKSortedInvariants) {
    std::mt19937 gen(12345);
    const std::size_t n = 500;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t k : {1UL, 10UL, 50UL, 250UL, n - 1}) {
        auto copy = data;
        algoat::selection::top_k_sorted(copy, k);

        // Prefix [0, k) must be strictly sorted
        EXPECT_TRUE(std::is_sorted(copy.begin(), copy.begin() + k));

        // Prefix must hold the first k sorted elements
        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_EQ(copy[i], static_cast<int>(i + 1));
        }

        // All elements in [0, k) must be <= elements in [k, n)
        if (k < n) {
            auto max_in_prefix = copy[k - 1];
            auto min_in_suffix = *std::min_element(copy.begin() + k, copy.end());
            EXPECT_LE(max_in_prefix, min_in_suffix);
        }
    }
}

TEST(TopKTest, CustomComparator) {
    std::vector<int> data = {1, 9, 2, 8, 3, 7, 4, 6, 5};
    const std::size_t k = 4;
    auto comp = std::greater<int>{};

    algoat::selection::top_k_sorted(data, k, comp);

    EXPECT_TRUE(std::is_sorted(data.begin(), data.begin() + k, comp));
    EXPECT_EQ(data[0], 9);
    EXPECT_EQ(data[1], 8);
    EXPECT_EQ(data[2], 7);
    EXPECT_EQ(data[3], 6);

    for (std::size_t i = k; i < data.size(); ++i) {
        EXPECT_LE(data[i], 6);
    }
}

TEST(TopKTest, CustomProjection) {
    std::vector<Student> students = {
        {"Alice", 50}, {"Bob", 10}, {"Charlie", 30}, {"Diana", 20}, {"Eve", 40}};
    const std::size_t k = 3;

    algoat::selection::top_k_sorted(students, k, {}, &Student::rank);

    EXPECT_EQ(students[0].rank, 10);
    EXPECT_EQ(students[1].rank, 20);
    EXPECT_EQ(students[2].rank, 30);
    EXPECT_GE(students[3].rank, 30);
    EXPECT_GE(students[4].rank, 30);
}

TEST(TopKTest, StdArraySupport) {
    std::array<int, 6> arr = {60, 10, 50, 20, 40, 30};
    algoat::selection::top_k_sorted(arr, 3);

    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 20);
    EXPECT_EQ(arr[2], 30);
}
