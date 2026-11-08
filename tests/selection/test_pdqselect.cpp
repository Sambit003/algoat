#include "algoat/selection/pdqselect.hpp"
#include "test_selection_common.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using algoat::selection::testing::verify_selection_invariant;

struct Item {
    int key;
    std::string value;

    auto operator<=>(const Item& other) const = default;
};

} // namespace

TEST(PdqSelectTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    auto res = algoat::selection::pdqselect(empty.begin(), empty.begin(), empty.end());
    EXPECT_EQ(res, empty.end());

    std::vector<int> single = {42};
    auto res_single = algoat::selection::pdqselect(single.begin(), single.begin(), single.end());
    EXPECT_EQ(res_single, single.end());
    EXPECT_EQ(single[0], 42);
}

TEST(PdqSelectTest, TwoElements) {
    {
        std::vector<int> data = {10, 5};
        auto res = algoat::selection::pdqselect(data.begin(), data.begin(), data.end());
        EXPECT_EQ(res, data.end());
        EXPECT_EQ(data[0], 5);
        EXPECT_EQ(data[1], 10);
        verify_selection_invariant(data, data.begin());
    }
    {
        std::vector<int> data = {10, 5};
        auto res = algoat::selection::pdqselect(data.begin(), data.begin() + 1, data.end());
        EXPECT_EQ(res, data.end());
        EXPECT_EQ(data[0], 5);
        EXPECT_EQ(data[1], 10);
        verify_selection_invariant(data, data.begin() + 1);
    }
}

TEST(PdqSelectTest, ExtremumMinAndMax) {
    std::mt19937 gen(777);
    const std::size_t n = 200;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    // Test min (k = 0)
    {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.begin(), copy.end());
        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy[0], 1);
        verify_selection_invariant(copy, copy.begin());
    }

    // Test max (k = n - 1)
    {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.end() - 1, copy.end());
        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy.back(), static_cast<int>(n));
        verify_selection_invariant(copy, copy.end() - 1);
    }
}

TEST(PdqSelectTest, ArbitraryPermutations) {
    std::mt19937 gen(1337);
    const std::size_t n = 350;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t k :
         std::initializer_list<std::size_t>{0, 1, 10, n / 4, n / 2, 3 * n / 4, n - 2, n - 1}) {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.begin() + k, copy.end());

        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy[k], static_cast<int>(k + 1));
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(PdqSelectTest, AllElementsIdentical) {
    const std::size_t n = 500;
    std::vector<int> data(n, 7);
    auto it = algoat::selection::pdqselect(data.begin(), data.begin() + n / 2, data.end());

    EXPECT_EQ(it, data.end());
    for (int v : data) {
        EXPECT_EQ(v, 7);
    }
    verify_selection_invariant(data, data.begin() + n / 2);
}

TEST(PdqSelectTest, ManyDuplicates) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 5);
    const std::size_t n = 1000;
    std::vector<int> data(n);
    for (auto& x : data) {
        x = dist(gen);
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    for (std::size_t k :
         std::initializer_list<std::size_t>{0, 10, n / 4, n / 2, 3 * n / 4, n - 1}) {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.begin() + k, copy.end());

        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy[k], expected[k]);
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(PdqSelectTest, OrganPipeDistribution) {
    const std::size_t n = 600;
    std::vector<int> data(n);
    for (std::size_t i = 0; i < n / 2; ++i) {
        data[i] = static_cast<int>(i);
        data[n - 1 - i] = static_cast<int>(i);
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    for (std::size_t k : std::initializer_list<std::size_t>{0, n / 4, n / 2, 3 * n / 4, n - 1}) {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.begin() + k, copy.end());
        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy[k], expected[k]);
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(PdqSelectTest, ReverseSortedDistribution) {
    const std::size_t n = 512;
    std::vector<int> data(n);
    std::iota(data.rbegin(), data.rend(), 1);

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    for (std::size_t k : std::initializer_list<std::size_t>{0, 1, n / 2, n - 2, n - 1}) {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.begin() + k, copy.end());
        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy[k], expected[k]);
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(PdqSelectTest, CustomComparator) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    const std::size_t k = 2;

    auto copy = data;
    auto it = algoat::selection::pdqselect(copy.begin(), copy.begin() + k, copy.end(),
                                           std::greater<int>{});
    EXPECT_EQ(it, copy.end());

    // With greater<int>, index 2 should be the 3rd largest element
    auto expected = data;
    std::sort(expected.begin(), expected.end(), std::greater<int>{});
    EXPECT_EQ(copy[k], expected[k]);

    verify_selection_invariant(copy, copy.begin() + k, std::greater<int>{});
}

TEST(PdqSelectTest, CustomProjection) {
    std::vector<Item> data = {{10, "ten"}, {2, "two"},   {8, "eight"},
                              {1, "one"},  {7, "seven"}, {5, "five"}};
    const std::size_t k = 3;

    auto it = algoat::selection::pdqselect(data.begin(), data.begin() + k, data.end(),
                                           std::less<>{}, &Item::key);
    EXPECT_EQ(it, data.end());

    EXPECT_EQ(data[k].key, 7);
    verify_selection_invariant(data, data.begin() + k, std::less<>{}, &Item::key);
}

TEST(PdqSelectTest, RangeOverload) {
    std::vector<int> data = {9, 2, 7, 4, 5, 6, 3, 8, 1};
    const std::size_t k = 4;

    auto it = algoat::selection::pdqselect(data, data.begin() + k);
    EXPECT_EQ(it, data.end());
    EXPECT_EQ(data[k], 5);
    verify_selection_invariant(data, data.begin() + k);
}

TEST(PdqSelectTest, SpanOverloadsDynamicAndFixedExtent) {
    // Dynamic extent
    {
        std::vector<int> data = {9, 8, 7, 6, 5, 4, 3, 2, 1};
        const std::size_t k = 3;
        algoat::selection::pdqselect(std::span<int>(data), k);
        EXPECT_EQ(data[k], 4);
        verify_selection_invariant(data, data.begin() + k);
    }
    // Fixed extent
    {
        std::array<int, 9> data = {9, 8, 7, 6, 5, 4, 3, 2, 1};
        const std::size_t k = 3;
        algoat::selection::pdqselect(std::span<int, 9>(data), k);
        EXPECT_EQ(data[k], 4);
        verify_selection_invariant(data, data.begin() + k);
    }
}

TEST(PdqSelectTest, LargeDatasetBlockPartitioning) {
    std::mt19937_64 rng(999);
    const std::size_t n = 100000;
    std::vector<int64_t> data(n);
    for (auto& x : data) {
        x = static_cast<int64_t>(rng());
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    for (std::size_t k :
         std::initializer_list<std::size_t>{0, 100, n / 4, n / 2, 3 * n / 4, n - 1}) {
        auto copy = data;
        auto it = algoat::selection::pdqselect(copy.begin(), copy.begin() + k, copy.end());
        EXPECT_EQ(it, copy.end());
        EXPECT_EQ(copy[k], expected[k]);
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(PdqSelectTest, MultiMillionElementStressEquivalence) {
    std::mt19937_64 rng(12345);
    const std::size_t n = 1000000;
    std::vector<int64_t> data(n);
    for (auto& x : data) {
        x = static_cast<int64_t>(rng());
    }

    const std::size_t k = n / 2;
    auto copy_pdq = data;
    auto copy_std = data;

    algoat::selection::pdqselect(copy_pdq.begin(), copy_pdq.begin() + k, copy_pdq.end());
    std::nth_element(copy_std.begin(), copy_std.begin() + k, copy_std.end());

    EXPECT_EQ(copy_pdq[k], copy_std[k]);
    verify_selection_invariant(copy_pdq, copy_pdq.begin() + k);
}

TEST(PdqSelectTest, ConstexprEvaluation) {
    constexpr auto res = []() {
        std::array<int, 7> arr = {5, 2, 6, 1, 3, 4, 7};
        algoat::selection::pdqselect(arr.begin(), arr.begin() + 3, arr.end());
        return arr;
    }();
    static_assert(res[3] == 4);
}
