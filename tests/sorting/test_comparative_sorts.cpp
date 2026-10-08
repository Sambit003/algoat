#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

namespace algoat::sorting::testing {
namespace {

template <typename AlgoType> class ComparativeSortTest : public ::testing::Test {
protected:
    AlgoType algo;

    template <typename T> void verify_sort(std::vector<T>& data) {
        std::vector<T> expected = data;
        std::sort(expected.begin(), expected.end());

        algo.sort(std::span<T>(data.data(), data.size()));

        EXPECT_EQ(data, expected);
        EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    }
};

class SortAlgoNameGenerator {
public:
    template <typename T> static std::string GetName(int /*i*/) {
        std::string name(T{}.name());
        if (!name.empty()) {
            name[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));
        }
        return name;
    }
};

using ComparativeSortAlgos =
    ::testing::Types<SelectionSort, BubbleSort, InsertionSort, ShellSort, CombSort, GnomeSort,
                     CycleSort, QuickSort, MergeSort, HeapSort, IntroSort, TimSort, BlockSort>;

TYPED_TEST_SUITE(ComparativeSortTest, ComparativeSortAlgos, SortAlgoNameGenerator);

TYPED_TEST(ComparativeSortTest, RandomData) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-100, 100);
    std::vector<int> data(100);
    for (int& x : data)
        x = dist(gen);
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, EmptyInput) {
    std::vector<int> data;
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, SingleElement) {
    std::vector<int> data = {42};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, TwoElementsSorted) {
    std::vector<int> data = {1, 2};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, TwoElementsReverse) {
    std::vector<int> data = {2, 1};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, TwoElementsEqual) {
    std::vector<int> data = {2, 2};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, AllDuplicates) {
    std::vector<int> data = {7, 7, 7, 7, 7, 7, 7};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, AlreadySorted) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, ReverseSorted) {
    std::vector<int> data = {8, 7, 6, 5, 4, 3, 2, 1};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, OrganPipe) {
    std::vector<int> data = {1, 2, 3, 4, 4, 3, 2, 1};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, FewUnique) {
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = i % 3;
    }
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, MixedSigns) {
    std::vector<int> data = {-5, 3, -1, 0, 4, -8, 2, 0};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, LargeRandom) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::vector<int> data(1000);
    for (int& x : data)
        x = dist(gen);
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, FloatingPointData) {
    std::vector<double> data = {3.14, -2.71, 0.0, -0.0, 1.618, -1.414, 2.71};
    this->verify_sort(data);
}

TYPED_TEST(ComparativeSortTest, StringData) {
    std::vector<std::string> data = {"apple", "banana", "cherry", "apple", "date", "elderberry"};
    this->verify_sort(data);
}

TEST(FreeFunctionSortTest, DefaultComparatorOverloads) {
    std::vector<int> sample = {5, 2, 8, 1, 9, 3, 7, 4, 6};

    auto test_free = [&](auto sort_fn) {
        auto copy = sample;
        sort_fn(std::span{copy});
        EXPECT_TRUE(std::is_sorted(copy.begin(), copy.end()));
    };

    test_free([](auto s) { quicksort(s); });
    test_free([](auto s) { heapsort(s); });
    test_free([](auto s) { mergesort(s); });
    test_free([](auto s) { insertionsort(s); });
    test_free([](auto s) { selectionsort(s); });
    test_free([](auto s) { bubblesort(s); });
    test_free([](auto s) { shellsort(s); });
    test_free([](auto s) { combsort(s); });
    test_free([](auto s) { gnomesort(s); });
    test_free([](auto s) { cyclesort(s); });
    test_free([](auto s) { introsort(s); });
    test_free([](auto s) { timsort(s); });
    test_free([](auto s) { blocksort(s); });
    test_free([](auto s) { countingsort(s); });
    test_free([](auto s) { bucketsort(s); });
    test_free([](auto s) { radixsort_lsd(s); });
    test_free([](auto s) { radixsort_msd(s); });
    test_free([](auto s) { radixsort_inplace_msd(s); });
    test_free([](auto s) { pigeonholesort(s); });
}

TEST(FreeFunctionSortTest, CustomComparatorOverloads) {
    std::vector<int> sample = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    auto greater_comp = std::greater<int>{};

    auto test_custom = [&](auto sort_fn) {
        auto copy = sample;
        sort_fn(std::span{copy}, greater_comp);
        EXPECT_TRUE(std::is_sorted(copy.begin(), copy.end(), greater_comp));
    };

    test_custom([](auto s, auto c) { quicksort(s, c); });
    test_custom([](auto s, auto c) { heapsort(s, c); });
    test_custom([](auto s, auto c) { mergesort(s, c); });
    test_custom([](auto s, auto c) { insertionsort(s, c); });
    test_custom([](auto s, auto c) { selectionsort(s, c); });
    test_custom([](auto s, auto c) { bubblesort(s, c); });
    test_custom([](auto s, auto c) { shellsort(s, c); });
    test_custom([](auto s, auto c) { combsort(s, c); });
    test_custom([](auto s, auto c) { gnomesort(s, c); });
    test_custom([](auto s, auto c) { cyclesort(s, c); });
    test_custom([](auto s, auto c) { introsort(s, c); });
    test_custom([](auto s, auto c) { blocksort(s, c); });
}

TEST(FreeFunctionSortTest, BitonicSortPowerOfTwo) {
    std::vector<int> sample = {8, 3, 7, 4, 2, 9, 1, 5};
    bitonicsort(std::span{sample});
    EXPECT_TRUE(std::is_sorted(sample.begin(), sample.end()));

    std::vector<int> bitonic_desc = {8, 3, 7, 4, 2, 9, 1, 5};
    bitonicsort(std::span{bitonic_desc}, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(bitonic_desc.begin(), bitonic_desc.end(), std::greater<int>{}));
}

} // namespace
} // namespace algoat::sorting::testing
