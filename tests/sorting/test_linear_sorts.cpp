#include "algoat/sorting/bucketsort.hpp"
#include "algoat/sorting/countingsort.hpp"
#include "algoat/sorting/pigeonholesort.hpp"
#include "algoat/sorting/radixsort.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

using namespace algoat::sorting;

template <typename Algo> class LinearSortTest : public ::testing::Test {
protected:
    Algo algo;
    void verify_sort(std::vector<int>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

using LinearAlgos =
    ::testing::Types<CountingSort, PigeonholeSort, RadixSortLSD, RadixSortMSD, BucketSort>;
TYPED_TEST_SUITE(LinearSortTest, LinearAlgos);

TYPED_TEST(LinearSortTest, RandomData) {
    std::vector<int> data = {5, -3, 8, 1, -9, 2, 7};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, EmptyInput) {
    std::vector<int> data;
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, SingleElement) {
    std::vector<int> data = {42};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, AllDuplicates) {
    std::vector<int> data = {7, 7, 7, 7, 7};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, AlreadySorted) {
    std::vector<int> data = {-5, -2, 0, 4, 5, 6, 7};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, ReverseSorted) {
    std::vector<int> data = {7, 6, 5, 4, 0, -2, -5};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, LargeRandom) {
    std::vector<int> data(1000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (int& x : data) {
        x = dist(gen);
    }
    this->verify_sort(data);
}

TEST(RadixSortTest, SignedIntegerBoundaries) {
    std::vector<int32_t> data = {
        std::numeric_limits<int32_t>::max(),     std::numeric_limits<int32_t>::min(),    0, -1, 1,
        std::numeric_limits<int32_t>::min() + 1, std::numeric_limits<int32_t>::max() - 1};
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::sorting::RadixSortLSD{}.sort(std::span{data});
    EXPECT_EQ(data, expected);
}

template <typename Algo> class RangeSortTest : public ::testing::Test {
protected:
    Algo algo;
    template <typename T> void verify_sort(std::vector<T>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

using RangeAlgos = ::testing::Types<CountingSort, PigeonholeSort, BucketSort>;
TYPED_TEST_SUITE(RangeSortTest, RangeAlgos);

TYPED_TEST(RangeSortTest, FullInt8Range) {
    std::vector<int8_t> data;
    for (int i = -128; i <= 127; ++i) {
        data.push_back(static_cast<int8_t>(i));
    }
    std::reverse(data.begin(), data.end());
    this->verify_sort(data);
}

TYPED_TEST(RangeSortTest, ClusteredNearMinAndMax) {
    constexpr int32_t min_v = std::numeric_limits<int32_t>::min();
    std::vector<int32_t> min_cluster = {min_v + 10, min_v, min_v + 5, min_v + 2, min_v};
    this->verify_sort(min_cluster);

    constexpr int32_t max_v = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> max_cluster = {max_v - 5, max_v, max_v - 10, max_v - 2, max_v};
    this->verify_sort(max_cluster);
}

template <typename Algo, typename T>
concept CanSort = requires(Algo a, std::span<T> arr) { a.sort(arr); };

TEST(AlgorithmConstraintTest, RejectsBoolAtCompileTime) {
    static_assert(!CanSort<algoat::sorting::CountingSort, bool>, "CountingSort must reject bool");
    static_assert(!CanSort<algoat::sorting::BucketSort, bool>, "BucketSort must reject bool");
    static_assert(!CanSort<algoat::sorting::PigeonholeSort, bool>,
                  "PigeonholeSort must reject bool");
    static_assert(!CanSort<algoat::sorting::RadixSortLSD, bool>, "RadixSortLSD must reject bool");
    static_assert(!CanSort<algoat::sorting::RadixSortMSD, bool>, "RadixSortMSD must reject bool");
    SUCCEED();
}
