#include "../sorting/test_stable_item.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::sorting;
using algoat::sorting::testing::StableItem;

namespace rc {
template <> struct Arbitrary<StableItem> {
    static Gen<StableItem> arbitrary() {
        return gen::build<StableItem>(gen::set(&StableItem::key, gen::arbitrary<int>()),
                                      gen::set(&StableItem::original_index, gen::just(-1)));
    }
};
} // namespace rc

template <typename Algo> class ComparativeSortPBT : public ::testing::Test {
protected:
    Algo algo;
};

using ComparativeSortAlgos =
    ::testing::Types<SelectionSort, BubbleSort, InsertionSort, ShellSort, CombSort, GnomeSort,
                     CycleSort, QuickSort, MergeSort, HeapSort, IntroSort, TimSort, BlockSort>;

TYPED_TEST_SUITE(ComparativeSortPBT, ComparativeSortAlgos);

TYPED_TEST(ComparativeSortPBT, Invariants) {
    rc::check("Comparative sort invariants", [this](std::vector<int> input) {
        auto original = input;
        this->algo.sort(std::span{input});

        // 1. Conservation
        auto expected = original;
        std::sort(expected.begin(), expected.end());
        RC_ASSERT(input.size() == expected.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        // 2. Monotonic Sortedness
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));

        // 3. Idempotency
        auto sorted_again = input;
        this->algo.sort(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

template <typename Algo> class StableSortPBT : public ::testing::Test {
protected:
    Algo algo;
};

using StableSortAlgos = ::testing::Types<BubbleSort, InsertionSort, MergeSort, TimSort, BlockSort>;

TYPED_TEST_SUITE(StableSortPBT, StableSortAlgos);

TYPED_TEST(StableSortPBT, StabilityInvariant) {
    rc::check("Stable sort invariants", [this](std::vector<int> input_vals) {
        std::vector<StableItem> data;
        for (size_t i = 0; i < input_vals.size(); ++i) {
            data.push_back({input_vals[i], static_cast<int>(i)});
        }

        this->algo.sort(std::span{data});

        RC_ASSERT(std::is_sorted(data.begin(), data.end()));

        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i - 1].key == data[i].key) {
                RC_ASSERT(data[i - 1].original_index < data[i].original_index);
            }
        }
    });
}

// Special case for BitonicSort (requires power of 2)
TEST(BitonicSortPBT, Invariants) {
    rc::check("Bitonic sort invariants", [](std::vector<int> input) {
        size_t n = 1;
        while (n < input.size())
            n *= 2;
        input.resize(n, 0);

        auto original = input;
        BitonicSort{}.sort(std::span{input});

        auto expected = original;
        std::sort(expected.begin(), expected.end());
        RC_ASSERT(input == expected);
    });
}

// Linear Sorts (Radix, Counting, Bucket, etc. taking integers)
template <typename Algo> class LinearSortPBT : public ::testing::Test {
protected:
    Algo algo;
};

using LinearSortAlgos = ::testing::Types<CountingSort, PigeonholeSort, RadixSortLSD, RadixSortMSD,
                                         RadixSortInPlaceMSD, BucketSort>;

TYPED_TEST_SUITE(LinearSortPBT, LinearSortAlgos);

TYPED_TEST(LinearSortPBT, Invariants) {
    rc::check("Linear sort invariants", [this]() {
        auto input = *rc::gen::container<std::vector<int>>(rc::gen::inRange(-10000, 10000));
        auto original = input;
        this->algo.sort(std::span{input});

        auto expected = original;
        std::sort(expected.begin(), expected.end());
        RC_ASSERT(input == expected);
    });
}
