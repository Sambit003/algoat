#include "algoat/selection/selection.hpp"
#include "test_selection_common.hpp"

#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>

namespace {

using algoat::selection::testing::verify_selection_invariant;

struct Record {
    int id;
    double score;

    auto operator<=>(const Record& other) const = default;
};

} // namespace

TEST(SelectOptimalTest, EmptyAndSingleElement) {
    std::vector<int> empty;
    auto res = algoat::selection::select_optimal(empty.begin(), empty.begin(), empty.end());
    EXPECT_EQ(res, empty.end());

    std::vector<int> single = {99};
    auto res_single =
        algoat::selection::select_optimal(single.begin(), single.begin(), single.end());
    EXPECT_EQ(res_single, single.end());
    EXPECT_EQ(single[0], 99);
}

TEST(SelectOptimalTest, TwoElements) {
    std::vector<int> data = {42, 17};
    algoat::selection::select_optimal(data.begin(), data.begin(), data.end());
    EXPECT_EQ(data[0], 17);
    EXPECT_EQ(data[1], 42);

    algoat::selection::select_optimal(data.begin(), data.begin() + 1, data.end());
    EXPECT_EQ(data[0], 17);
    EXPECT_EQ(data[1], 42);
}

TEST(SelectOptimalTest, SmallInputs) {
    std::mt19937 gen(101);
    const std::size_t n = 150;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t k : std::initializer_list<std::size_t>{0, 1, n / 4, n / 2, 3 * n / 4, n - 1}) {
        auto copy = data;
        algoat::selection::select_optimal(copy.begin(), copy.begin() + k, copy.end());
        EXPECT_EQ(copy[k], static_cast<int>(k + 1));
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(SelectOptimalTest, LargeInputsFloydRivestSampling) {
    // Tests n > 600 where recursive Floyd-Rivest sampling triggers
    std::mt19937 gen(2024);
    const std::size_t n = 5000;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    for (std::size_t k : std::initializer_list<std::size_t>{0, 10, n / 10, n / 2, n - 100, n - 1}) {
        auto copy = data;
        algoat::selection::select_optimal(copy.begin(), copy.begin() + k, copy.end());
        EXPECT_EQ(copy[k], static_cast<int>(k + 1));
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(SelectOptimalTest, VeryLargeDataset) {
    std::mt19937 gen(777);
    const std::size_t n = 50000;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    const std::size_t k = n / 2;
    algoat::selection::select_optimal(data.begin(), data.begin() + k, data.end());
    EXPECT_EQ(data[k], static_cast<int>(k + 1));
    verify_selection_invariant(data, data.begin() + k);
}

TEST(SelectOptimalTest, AllElementsIdentical) {
    const std::size_t n = 1500;
    std::vector<int> data(n, 42);
    const std::size_t k = n / 2;

    algoat::selection::select_optimal(data.begin(), data.begin() + k, data.end());
    for (int v : data) {
        EXPECT_EQ(v, 42);
    }
    verify_selection_invariant(data, data.begin() + k);
}

TEST(SelectOptimalTest, HighDuplicateFrequency) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 10);
    const std::size_t n = 3000;
    std::vector<int> data(n);
    for (auto& x : data) {
        x = dist(gen);
    }

    auto expected = data;
    std::sort(expected.begin(), expected.end());

    for (std::size_t k : std::initializer_list<std::size_t>{0, 100, n / 2, n - 100, n - 1}) {
        auto copy = data;
        algoat::selection::select_optimal(copy.begin(), copy.begin() + k, copy.end());
        EXPECT_EQ(copy[k], expected[k]);
        verify_selection_invariant(copy, copy.begin() + k);
    }
}

TEST(SelectOptimalTest, CustomComparator) {
    std::mt19937 gen(888);
    const std::size_t n = 1200;
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    std::shuffle(data.begin(), data.end(), gen);

    auto comp = std::greater<int>{};
    const std::size_t k = n / 3;

    algoat::selection::select_optimal(data.begin(), data.begin() + k, data.end(), comp);

    auto expected = data;
    std::sort(expected.begin(), expected.end(), comp);
    EXPECT_EQ(data[k], expected[k]);
    verify_selection_invariant(data, data.begin() + k, comp);
}

TEST(SelectOptimalTest, CustomProjection) {
    const std::size_t n = 1000;
    std::vector<Record> records;
    records.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        records.push_back({static_cast<int>(i), static_cast<double>(n - i)});
    }
    std::mt19937 gen(999);
    std::shuffle(records.begin(), records.end(), gen);

    const std::size_t k = 250;
    algoat::selection::select_optimal(records.begin(), records.begin() + k, records.end(), {},
                                      &Record::id);

    EXPECT_EQ(records[k].id, static_cast<int>(k));
    verify_selection_invariant(records, records.begin() + k, {}, &Record::id);
}

TEST(SelectOptimalTest, RangeOverload) {
    std::vector<int> data = {15, 3, 2, 8, 7, 6, 4, 1, 9, 12};
    const std::size_t k = 3;

    algoat::selection::select_optimal(data, data.begin() + k);
    EXPECT_EQ(data[k], 4);
    verify_selection_invariant(data, data.begin() + k);
}

TEST(SelectOptimalTest, ConstexprEvaluation) {
    constexpr auto test_constexpr = []() {
        std::array<int, 7> arr = {5, 2, 7, 1, 6, 3, 4};
        algoat::selection::select_optimal(arr.begin(), arr.begin() + 3, arr.end());
        return arr[3];
    };
    static_assert(test_constexpr() == 4);
}
