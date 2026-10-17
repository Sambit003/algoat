#include "algoat/selection/selection.hpp"
#include "test_selection_common.hpp"

#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

namespace {

using algoat::selection::testing::verify_selection_invariant;

struct Item {
    int key;
    std::string value;

    auto operator<=>(const Item& other) const = default;
};

} // namespace

TEST(NthElementTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    auto res = algoat::selection::nth_element(empty.begin(), empty.begin(), empty.end());
    EXPECT_EQ(res, empty.end());

    std::vector<int> single = {42};
    auto res_single = algoat::selection::nth_element(single.begin(), single.begin(), single.end());
    EXPECT_EQ(res_single, single.end());
    EXPECT_EQ(single[0], 42);
}

TEST(NthElementTest, TwoElements) {
    {
        std::vector<int> data = {10, 5};
        algoat::selection::nth_element(data.begin(), data.begin(), data.end());
        EXPECT_EQ(data[0], 5);
        EXPECT_EQ(data[1], 10);
    }
    {
        std::vector<int> data = {10, 5};
        algoat::selection::nth_element(data.begin(), data.begin() + 1, data.end());
        EXPECT_EQ(data[0], 5);
        EXPECT_EQ(data[1], 10);
    }
}

TEST(NthElementTest, ArbitraryPermutations) {
    std::mt19937 gen(1337);
    const std::size_t n = 250;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t k : {0UL, 1UL, n / 4, n / 2, 3 * n / 4, n - 2, n - 1}) {
        auto copy = data;
        algoat::selection::nth_element(copy.begin(), copy.begin() + k, copy.end());

        EXPECT_EQ(copy[k], static_cast<int>(k + 1));
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(NthElementTest, AllElementsIdentical) {
    const std::size_t n = 500;
    std::vector<int> data(n, 7);
    algoat::selection::nth_element(data.begin(), data.begin() + n / 2, data.end());

    for (int v : data) {
        EXPECT_EQ(v, 7);
    }
    verify_selection_invariant(data, data.begin() + n / 2);
}

TEST(NthElementTest, ManyDuplicates) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 5);
    const std::size_t n = 600;
    std::vector<int> data(n);
    for (auto& x : data) {
        x = dist(gen);
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    for (std::size_t k : {0UL, 10UL, n / 3, n / 2, 2 * n / 3, n - 1}) {
        auto copy = data;
        algoat::selection::nth_element(copy.begin(), copy.begin() + k, copy.end());

        EXPECT_EQ(copy[k], expected[k]);
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(NthElementTest, CustomComparator) {
    std::vector<int> data = {9, 2, 8, 3, 7, 1, 4, 6, 5};
    auto comp = std::greater<int>{};
    const std::size_t k = 3;

    algoat::selection::nth_element(data.begin(), data.begin() + k, data.end(), comp);

    auto expected = data;
    std::sort(expected.begin(), expected.end(), comp);
    EXPECT_EQ(data[k], expected[k]);
    verify_selection_invariant(data, data.begin() + k, comp);
}

TEST(NthElementTest, CustomProjection) {
    std::vector<Item> items = {{9, "nine"}, {2, "two"},  {8, "eight"}, {3, "three"}, {7, "seven"},
                               {1, "one"},  {4, "four"}, {6, "six"},   {5, "five"}};
    auto proj = &Item::key;
    const std::size_t k = 4;

    algoat::selection::nth_element(items.begin(), items.begin() + k, items.end(), {}, proj);

    EXPECT_EQ(items[k].key, 5);
    verify_selection_invariant(items, items.begin() + k, {}, proj);
}

TEST(NthElementTest, RangeOverload) {
    std::vector<int> data = {9, 1, 8, 2, 7, 3, 6, 4, 5};
    const std::size_t k = 4;

    algoat::selection::nth_element(data, data.begin() + k);

    EXPECT_EQ(data[k], 5);
    verify_selection_invariant(data, data.begin() + k);
}

TEST(NthElementTest, AdversarialKillerSequence) {
    // Organ Pipe / Median-of-3 killer sequence:
    // [0, 2, 4, 6, ..., 2k, ..., 7, 5, 3, 1]
    const std::size_t n = 2000;
    std::vector<int> data;
    data.reserve(n);
    for (std::size_t i = 0; i < n; i += 2) {
        data.push_back(static_cast<int>(i));
    }
    for (long long i = (n % 2 == 0 ? static_cast<long long>(n) - 1 : static_cast<long long>(n) - 2);
         i >= 1; i -= 2) {
        data.push_back(static_cast<int>(i));
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    const std::size_t k = n / 2;
    algoat::selection::nth_element(data.begin(), data.begin() + k, data.end());

    EXPECT_EQ(data[k], expected[k]);
    verify_selection_invariant(data, data.begin() + k);
}
