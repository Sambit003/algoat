#include "../sorting/test_stable_item.hpp"
#include "algoat/algoat.hpp"
#include "algoat/numerics/float16_sort.hpp"
#include "algoat/numerics/hilbert.hpp"
#include "algoat/numerics/hybrid_sfc.hpp"
#include "algoat/numerics/morton.hpp"
#include "algoat/sorting/boolean_sort.hpp"
#include "algoat/sorting/hilbert_sort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::sorting;
using algoat::sorting::testing::StableItem;
namespace p = algoat::meta::policy;

template <typename Algo> class ComparativeSortPBT : public ::testing::Test {
protected:
    Algo algo;
};

using ComparativeSortAlgos =
    ::testing::Types<SelectionSort, BubbleSort, InsertionSort, ShellSort, CombSort, GnomeSort,
                     CycleSort, QuickSort, MergeSort, HeapSort, IntroSort, TimSort, BlockSort,
                     PdqSort, CrumSort, FluxSort>;

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

using StableSortAlgos =
    ::testing::Types<BubbleSort, InsertionSort, MergeSort, TimSort, BlockSort, FluxSort>;

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
        RC_ASSERT(input.size() == expected.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));
        auto sorted_again = input;
        BitonicSort{}.sort(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
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
        RC_ASSERT(input.size() == expected.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));
        auto sorted_again = input;
        this->algo.sort(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

TEST(HilbertSortPBT, Invariants) {
    rc::check("Hilbert Radix sort invariants (Complex Float)", []() {
        auto input_pairs = *rc::gen::container<std::vector<std::pair<int, int>>>(
            rc::gen::pair(rc::gen::inRange(-1000, 1000), rc::gen::inRange(-1000, 1000)));
        std::vector<std::complex<float>> input;
        for (const auto& p : input_pairs) {
            input.emplace_back(p.first, p.second);
        }

        auto original = input;
        HilbertSort{}.sort(std::span{input});

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_sorted(input.begin(), input.end(), algoat::numerics::HilbertCompare{}));

        auto sorted_again = input;
        HilbertSort{}.sort(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

TEST(HybridSFCSortPBT, Invariants) {
    rc::check("Hybrid SFC Radix sort invariants (Complex Float)", []() {
        auto input_pairs = *rc::gen::container<std::vector<std::pair<int, int>>>(
            rc::gen::pair(rc::gen::inRange(-1000, 1000), rc::gen::inRange(-1000, 1000)));
        std::vector<std::complex<float>> input;
        for (const auto& p : input_pairs) {
            input.emplace_back(static_cast<float>(p.first), static_cast<float>(p.second));
        }

        auto original = input;
        algoat::numerics::sort_complex_hybrid(std::span{input});

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_sorted(input.begin(), input.end(), algoat::numerics::HybridCompare{}));

        auto sorted_again = input;
        algoat::numerics::sort_complex_hybrid(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

TEST(MortonSortPBT, Invariants) {
    rc::check("Morton Radix sort invariants (Complex Float)", []() {
        auto input_pairs = *rc::gen::container<std::vector<std::pair<int, int>>>(
            rc::gen::pair(rc::gen::inRange(-1000, 1000), rc::gen::inRange(-1000, 1000)));
        std::vector<std::complex<float>> input;
        for (const auto& p : input_pairs) {
            input.emplace_back(static_cast<float>(p.first), static_cast<float>(p.second));
        }

        auto original = input;
        algoat::numerics::sort_complex_morton(std::span{input});

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_sorted(input.begin(), input.end(), algoat::numerics::MortonCompare{}));

        auto sorted_again = input;
        algoat::numerics::sort_complex_morton(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

TEST(BooleanSortPBT, BooleanSpanInvariants) {
    rc::check("Boolean span sort invariants", [](std::vector<uint8_t> raw_bits) {
        const size_t n = raw_bits.size();
        auto bool_buf = std::make_unique<bool[]>(n);
        size_t expected_true = 0;
        for (size_t i = 0; i < n; ++i) {
            bool_buf[i] = (raw_bits[i] % 2 != 0);
            if (bool_buf[i]) {
                ++expected_true;
            }
        }

        std::span<bool> span(bool_buf.get(), n);
        algoat::sorting::sort_boolean(span);

        RC_ASSERT(std::is_sorted(span.begin(), span.end()));
        size_t actual_true = 0;
        for (size_t i = 0; i < n; ++i) {
            if (span[i]) {
                ++actual_true;
            }
        }
        RC_ASSERT(actual_true == expected_true);
    });
}

TEST(BooleanSortPBT, IntegerZeroPartitionInvariants) {
    rc::check("Integer boolean sort zero-partition invariants", [](std::vector<int> input) {
        auto original = input;
        algoat::sorting::sort_boolean(std::span<int>{input});

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        bool seen_nonzero = false;
        for (int val : input) {
            if (val != 0) {
                seen_nonzero = true;
            } else {
                RC_ASSERT(!seen_nonzero);
            }
        }
    });
}

TEST(Float16SortPBT, Invariants) {
    rc::check("Float16 sort invariants", [](std::vector<uint16_t> input) {
        auto original = input;
        algoat::numerics::sort_float16(std::span<uint16_t>{input});

        auto expected = original;
        std::sort(expected.begin(), expected.end(), algoat::numerics::Float16Compare{});

        RC_ASSERT(input == expected);

        auto sorted_again = input;
        algoat::numerics::sort_float16(std::span<uint16_t>{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

TEST(StaticSortPBT, DefaultPolicyInvariants) {
    rc::check("Static sort default policy invariants", [](std::vector<int> input) {
        auto original = input;
        algoat::meta::static_sort(std::span{input});

        auto expected = original;
        std::sort(expected.begin(), expected.end());
        RC_ASSERT(input.size() == expected.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));

        auto sorted_again = input;
        algoat::meta::static_sort(std::span{sorted_again});
        RC_ASSERT(input == sorted_again);
    });
}

TEST(StaticSortPBT, StableInPlaceInvariants) {
    rc::check("Static sort stable in-place invariants", [](std::vector<int> input_vals) {
        std::vector<StableItem> data;
        for (size_t i = 0; i < input_vals.size(); ++i) {
            data.push_back({input_vals[i], static_cast<int>(i)});
        }

        algoat::meta::static_sort(std::span{data}, p::stable | p::in_place);

        RC_ASSERT(std::is_sorted(data.begin(), data.end()));
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i - 1].key == data[i].key) {
                RC_ASSERT(data[i - 1].original_index < data[i].original_index);
            }
        }
    });
}

TEST(StaticSortPBT, ForceRadixInvariants) {
    rc::check("Static sort force_radix invariants", []() {
        auto input = *rc::gen::container<std::vector<int>>(rc::gen::inRange(-10000, 10000));
        auto original = input;
        algoat::meta::static_sort(std::span{input}, p::force_radix);

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));
    });
}

TEST(StaticSortPBT, FixedExtentSortingNetworks) {
    rc::check("Fixed extent 4-element sorting network", [](std::array<int, 4> input) {
        auto original = input;
        algoat::meta::static_sort(std::span<int, 4>{input});
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
    });

    rc::check("Fixed extent 8-element sorting network", [](std::array<int, 8> input) {
        auto original = input;
        algoat::meta::static_sort(std::span<int, 8>{input});
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
    });

    rc::check("Fixed extent 16-element sorting network", [](std::array<int, 16> input) {
        auto original = input;
        algoat::meta::static_sort(std::span<int, 16>{input});
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));
    });
}

TEST(PdqSortPBT, CustomComparatorAndProjectionInvariants) {
    struct TestItem {
        int key;
        std::string payload;
        bool operator==(const TestItem& other) const = default;
    };

    rc::check("PdqSort with projection and descending comparator", [](std::vector<int> keys) {
        std::vector<TestItem> items;
        items.reserve(keys.size());
        for (int k : keys) {
            items.push_back({k, std::to_string(k)});
        }
        auto original_items = items;
        algoat::sorting::pdqsort(items, std::greater<int>{}, &TestItem::key);

        RC_ASSERT(items.size() == original_items.size());
        RC_ASSERT(
            std::is_sorted(items.begin(), items.end(),
                           [](const TestItem& a, const TestItem& b) { return a.key > b.key; }));
        RC_ASSERT(std::is_permutation(items.begin(), items.end(), original_items.begin(),
                                      original_items.end()));
    });
}
