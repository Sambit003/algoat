#include "../sorting/test_stable_item.hpp"
#include "algoat/algoat.hpp"
#include "algoat/numerics/float16_sort.hpp"
#include "algoat/numerics/hilbert.hpp"
#include "algoat/numerics/hybrid_sfc.hpp"
#include "algoat/numerics/morton.hpp"
#include "algoat/sorting/boolean_sort.hpp"
#include "algoat/sorting/hilbert_sort.hpp"
#include "algoat/sorting/sorting.hpp"
#include "algoat/strings/burstsort.hpp"
#include "algoat/strings/multikey_quicksort.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
                     PdqSort, CrumSort, FluxSort, Ips4oSort, VQSort, GrailSort, SmoothSort,
                     PatienceSort>;

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
    ::testing::Types<BubbleSort, InsertionSort, MergeSort, TimSort, BlockSort, FluxSort, GrailSort>;

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
                                         RadixSortInPlaceMSD, BucketSort, SkaSort>;

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

TEST(Ips4oSortPBT, CustomComparatorAndProjectionInvariants) {
    struct TestItem {
        int key;
        std::string payload;
        bool operator==(const TestItem& other) const = default;
    };

    rc::check("IPS4O with projection and descending comparator", [](std::vector<int> keys) {
        std::vector<TestItem> items;
        items.reserve(keys.size());
        for (int k : keys) {
            items.push_back({k, std::to_string(k)});
        }
        auto original_items = items;
        algoat::sorting::ips4o_parallel_sort(items, std::greater<int>{}, &TestItem::key);

        RC_ASSERT(items.size() == original_items.size());
        RC_ASSERT(
            std::is_sorted(items.begin(), items.end(),
                           [](const TestItem& a, const TestItem& b) { return a.key > b.key; }));
        RC_ASSERT(std::is_permutation(items.begin(), items.end(), original_items.begin(),
                                      original_items.end()));
    });
}

TEST(Ips4oSortPBT, VaryingSizesAndDuplicates) {
    rc::check("IPS4O with varying duplicate densities and block sizes", []() {
        const auto size = *rc::gen::inRange<std::size_t>(0, 5000);
        const auto max_val = *rc::gen::inRange<int>(1, 20);
        auto input = *rc::gen::container<std::vector<int>>(size, rc::gen::inRange<int>(0, max_val));

        auto original = input;
        algoat::sorting::ips4o_parallel_sort(input);

        // 1. Conservation
        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        // 2. Monotonic Sortedness
        RC_ASSERT(std::is_sorted(input.begin(), input.end()));

        // 3. Idempotency
        auto sorted_again = input;
        algoat::sorting::ips4o_parallel_sort(sorted_again);
        RC_ASSERT(input == sorted_again);
    });
}

TEST(StringsSortingPBT, MultiKeyQuicksortInvariants) {
    rc::check("MultiKeyQuicksort conservation, sortedness, and idempotency",
              [](std::vector<std::string> input) {
                  auto original = input;
                  algoat::strings::multikey_quicksort(std::span<std::string>{input});

                  // 1. Conservation
                  auto expected = original;
                  std::sort(expected.begin(), expected.end());
                  RC_ASSERT(input.size() == expected.size());
                  RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

                  // 2. Monotonic Sortedness
                  RC_ASSERT(std::is_sorted(input.begin(), input.end()));

                  // 3. Idempotency
                  auto sorted_again = input;
                  algoat::strings::multikey_quicksort(std::span<std::string>{sorted_again});
                  RC_ASSERT(input == sorted_again);
              });
}

TEST(StringsSortingPBT, BurstsortInvariants) {
    rc::check("Burstsort conservation, sortedness, and idempotency",
              [](std::vector<std::string> input) {
                  auto original = input;
                  algoat::strings::burstsort(std::span<std::string>{input});

                  // 1. Conservation
                  auto expected = original;
                  std::sort(expected.begin(), expected.end());
                  RC_ASSERT(input.size() == expected.size());
                  RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

                  // 2. Monotonic Sortedness
                  RC_ASSERT(std::is_sorted(input.begin(), input.end()));

                  // 3. Idempotency
                  auto sorted_again = input;
                  algoat::strings::burstsort(std::span<std::string>{sorted_again});
                  RC_ASSERT(input == sorted_again);
              });
}

TEST(VQSortPBT, CustomComparatorAndProjectionInvariants) {
    struct TestItem {
        int key;
        std::string payload;
        bool operator==(const TestItem& other) const = default;
    };

    rc::check("VQSort with projection and descending comparator", [](std::vector<int> keys) {
        std::vector<TestItem> items;
        items.reserve(keys.size());
        for (int k : keys) {
            items.push_back({k, std::to_string(k)});
        }
        auto original_items = items;
        algoat::sorting::vqsort(items, std::greater<int>{}, &TestItem::key);

        RC_ASSERT(items.size() == original_items.size());
        RC_ASSERT(
            std::is_sorted(items.begin(), items.end(),
                           [](const TestItem& a, const TestItem& b) { return a.key > b.key; }));
        RC_ASSERT(std::is_permutation(items.begin(), items.end(), original_items.begin(),
                                      original_items.end()));
    });
}

TEST(SkaSortPBT, ProjectionAndCompoundInvariants) {
    struct TestItem {
        int key;
        double priority;
        bool operator==(const TestItem& other) const = default;
    };

    rc::check("SkaSort with projection and ranges", [](std::vector<int> keys) {
        std::vector<TestItem> items;
        items.reserve(keys.size());
        for (int k : keys) {
            items.push_back({k, static_cast<double>(k) * 1.5});
        }
        auto original_items = items;
        algoat::sorting::ska_sort(items, &TestItem::key);

        RC_ASSERT(items.size() == original_items.size());
        RC_ASSERT(
            std::is_sorted(items.begin(), items.end(),
                           [](const TestItem& a, const TestItem& b) { return a.key < b.key; }));
        RC_ASSERT(std::is_permutation(items.begin(), items.end(), original_items.begin(),
                                      original_items.end()));
    });
}

TEST(GrailSortPBT, CustomComparatorAndProjectionInvariants) {
    struct TestItem {
        int key;
        int original_index;
        bool operator==(const TestItem& other) const = default;
    };

    rc::check("GrailSort with projection and descending comparator", [](std::vector<int> keys) {
        std::vector<TestItem> items;
        items.reserve(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            items.push_back({keys[i], static_cast<int>(i)});
        }
        auto original_items = items;
        algoat::sorting::grailsort(items, std::ranges::greater{}, &TestItem::key);

        RC_ASSERT(items.size() == original_items.size());
        RC_ASSERT(
            std::is_sorted(items.begin(), items.end(),
                           [](const TestItem& a, const TestItem& b) { return a.key > b.key; }));
        for (std::size_t i = 1; i < items.size(); ++i) {
            if (items[i - 1].key == items[i].key) {
                RC_ASSERT(items[i - 1].original_index < items[i].original_index);
            }
        }
    });

    rc::check("GrailSort with bounded unique keys", [](const std::vector<int>& raw) {
        std::vector<int> keys;
        keys.reserve(raw.size());
        for (int x : raw) {
            keys.push_back(std::abs(x) % 5);
        }
        std::vector<int> data = keys;
        std::vector<int> expected = keys;
        std::stable_sort(expected.begin(), expected.end());

        algoat::sorting::grailsort(std::span<int>{data});
        RC_ASSERT(data == expected);
    });
}

namespace {

template <std::size_t N, typename T> void check_simd_network_pbt(const std::string& label) {
    rc::check(label, [](std::array<T, N> input) {
        if constexpr (std::is_floating_point_v<T>) {
            for (T& v : input) {
                if (!std::isfinite(v)) {
                    v = static_cast<T>(0);
                }
            }
        }
        auto expected = input;
        std::sort(expected.begin(), expected.end());

        algoat::meta::SIMDSortingNetwork<N, T>::sort(std::span<T, N>{input});
        RC_ASSERT(input == expected);

        auto idempotent = input;
        algoat::meta::sort_simd_network(std::span<T, N>{idempotent});
        RC_ASSERT(idempotent == input);
    });
}

} // namespace

TEST(SimdSortingNetworkPBT, AllSizesAndTypesInvariants) {
    check_simd_network_pbt<8, std::int32_t>("SIMD network N=8 int32");
    check_simd_network_pbt<16, std::int32_t>("SIMD network N=16 int32");
    check_simd_network_pbt<32, std::int32_t>("SIMD network N=32 int32");
    check_simd_network_pbt<64, std::int32_t>("SIMD network N=64 int32");

    check_simd_network_pbt<8, std::uint32_t>("SIMD network N=8 uint32");
    check_simd_network_pbt<16, std::uint32_t>("SIMD network N=16 uint32");
    check_simd_network_pbt<32, std::uint32_t>("SIMD network N=32 uint32");
    check_simd_network_pbt<64, std::uint32_t>("SIMD network N=64 uint32");

    check_simd_network_pbt<8, float>("SIMD network N=8 float");
    check_simd_network_pbt<16, float>("SIMD network N=16 float");
    check_simd_network_pbt<32, float>("SIMD network N=32 float");
    check_simd_network_pbt<64, float>("SIMD network N=64 float");

    check_simd_network_pbt<8, std::int64_t>("SIMD network N=8 int64");
    check_simd_network_pbt<16, std::int64_t>("SIMD network N=16 int64");
    check_simd_network_pbt<32, std::int64_t>("SIMD network N=32 int64");
    check_simd_network_pbt<64, std::int64_t>("SIMD network N=64 int64");

    check_simd_network_pbt<8, std::uint64_t>("SIMD network N=8 uint64");
    check_simd_network_pbt<16, std::uint64_t>("SIMD network N=16 uint64");
    check_simd_network_pbt<32, std::uint64_t>("SIMD network N=32 uint64");
    check_simd_network_pbt<64, std::uint64_t>("SIMD network N=64 uint64");

    check_simd_network_pbt<8, double>("SIMD network N=8 double");
    check_simd_network_pbt<16, double>("SIMD network N=16 double");
    check_simd_network_pbt<32, double>("SIMD network N=32 double");
    check_simd_network_pbt<64, double>("SIMD network N=64 double");

    check_simd_network_pbt<16, std::int16_t>("SIMD network scalar fallback N=16 int16");

    rc::check("Fixed extent 32 and 64 static_sort integration",
              [](std::array<std::int32_t, 32> a32, std::array<double, 64> a64) {
                  auto exp32 = a32;
                  std::sort(exp32.begin(), exp32.end());
                  algoat::meta::static_sort(std::span<std::int32_t, 32>{a32});
                  RC_ASSERT(a32 == exp32);

                  for (double& d : a64) {
                      if (!std::isfinite(d)) {
                          d = 0.0;
                      }
                  }
                  auto exp64 = a64;
                  std::sort(exp64.begin(), exp64.end());
                  algoat::meta::static_sort(std::span<double, 64>{a64});
                  RC_ASSERT(a64 == exp64);
              });
}
