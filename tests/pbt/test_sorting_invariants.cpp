#include "algoat/sorting/bubblesort.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/selectionsort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <set>
#include <vector>

using namespace algoat::sorting;

// We will test core invariants using RapidCheck property-based tests.
// 1. Conservation
// 2. Monotonic Sortedness
// 3. Idempotency
// 4. Strict Stability

struct StableItem {
    int value;
    int original_index;
    auto operator<=>(const StableItem& other) const {
        return value <=> other.value;
    }
    bool operator==(const StableItem& other) const {
        return value == other.value;
    }
};

namespace rc {
template <> struct Arbitrary<StableItem> {
    static Gen<StableItem> arbitrary() {
        return gen::build<StableItem>(gen::set(&StableItem::value, gen::arbitrary<int>()),
                                      gen::set(&StableItem::original_index, gen::just(-1)));
    }
};
} // namespace rc

template <typename Algo> void verify_invariants(Algo& algo, std::vector<int> input) {
    auto original = input;
    algo.sort(std::span{input});

    // 1. Conservation (same elements)
    auto expected = original;
    std::sort(expected.begin(), expected.end());
    RC_ASSERT(input.size() == expected.size());
    RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

    // 2. Monotonic Sortedness
    RC_ASSERT(std::is_sorted(input.begin(), input.end()));

    // 3. Idempotency
    auto sorted_again = input;
    algo.sort(std::span{sorted_again});
    RC_ASSERT(input == sorted_again);
}

template <typename Algo> void verify_stability(Algo& algo, std::vector<int> input_vals) {
    std::vector<StableItem> data;
    for (size_t i = 0; i < input_vals.size(); ++i) {
        data.push_back({input_vals[i], static_cast<int>(i)});
    }

    algo.sort(std::span{data});

    RC_ASSERT(std::is_sorted(data.begin(), data.end()));

    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i - 1].value == data[i].value) {
            RC_ASSERT(data[i - 1].original_index < data[i].original_index);
        }
    }
}

// RapidCheck tests for QuickSort
RC_GTEST_PROP(SortingInvariantsPBT, QuickSort_Invariants, (std::vector<int> data)) {
    QuickSort algo;
    verify_invariants(algo, data);
}

// RapidCheck tests for MergeSort
RC_GTEST_PROP(SortingInvariantsPBT, MergeSort_Invariants, (std::vector<int> data)) {
    MergeSort algo;
    verify_invariants(algo, data);
}
RC_GTEST_PROP(SortingInvariantsPBT, MergeSort_Stability, (std::vector<int> data)) {
    MergeSort algo;
    verify_stability(algo, data);
}

// RapidCheck tests for HeapSort
RC_GTEST_PROP(SortingInvariantsPBT, HeapSort_Invariants, (std::vector<int> data)) {
    HeapSort algo;
    verify_invariants(algo, data);
}

// RapidCheck tests for InsertionSort
RC_GTEST_PROP(SortingInvariantsPBT, InsertionSort_Invariants, (std::vector<int> data)) {
    InsertionSort algo;
    verify_invariants(algo, data);
}
RC_GTEST_PROP(SortingInvariantsPBT, InsertionSort_Stability, (std::vector<int> data)) {
    InsertionSort algo;
    verify_stability(algo, data);
}

// BubbleSort
RC_GTEST_PROP(SortingInvariantsPBT, BubbleSort_Invariants, (std::vector<int> data)) {
    BubbleSort algo;
    verify_invariants(algo, data);
}
RC_GTEST_PROP(SortingInvariantsPBT, BubbleSort_Stability, (std::vector<int> data)) {
    BubbleSort algo;
    verify_stability(algo, data);
}
