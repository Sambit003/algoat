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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

using namespace algoat::sorting;
namespace p = algoat::meta::policy;

struct StableItem {
    int key;
    int original_index;
    auto operator<=>(const StableItem& other) const {
        return key <=> other.key;
    }
    bool operator==(const StableItem& other) const {
        return key == other.key;
    }
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0 || size % sizeof(int) != 0)
        return 0;
    size_t num_elements = size / sizeof(int);

    std::vector<int> original_ints(num_elements);
    std::memcpy(original_ints.data(), data, size);

    std::vector<StableItem> original;
    original.reserve(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        original.push_back({original_ints[i], static_cast<int>(i)});
    }

    std::vector<StableItem> expected = original;
    std::sort(expected.begin(), expected.end(), [](const StableItem& a, const StableItem& b) {
        if (a.key != b.key)
            return a.key < b.key;
        return a.original_index < b.original_index;
    });

    std::vector<StableItem> stable_expected = original;
    std::stable_sort(stable_expected.begin(), stable_expected.end(),
                     [](const StableItem& a, const StableItem& b) { return a.key < b.key; });

    auto test_algo = [&](auto algo, bool is_stable) {
        std::vector<StableItem> test_data = original;
        algo.sort(std::span{test_data});
        if (is_stable) {
            for (size_t i = 0; i < test_data.size(); ++i) {
                if (test_data[i].key != stable_expected[i].key ||
                    test_data[i].original_index != stable_expected[i].original_index)
                    __builtin_trap();
            }
        } else {
            std::vector<int> extracted_keys(test_data.size());
            for (size_t i = 0; i < test_data.size(); ++i)
                extracted_keys[i] = test_data[i].key;

            std::vector<int> exp_keys(stable_expected.size());
            for (size_t i = 0; i < stable_expected.size(); ++i)
                exp_keys[i] = stable_expected[i].key;

            if (extracted_keys != exp_keys)
                __builtin_trap();
        }
    };

    // Fast O(N log N) comparison algorithms
    test_algo(QuickSort{}, false);
    test_algo(HeapSort{}, false);
    test_algo(MergeSort{}, true);
    test_algo(IntroSort{}, false);
    test_algo(PdqSort{}, false);
    test_algo(TimSort{}, true);
    test_algo(BlockSort{}, true);
    test_algo(ShellSort{}, false);
    test_algo(CombSort{}, false);

    // Free function pdqsort
    {
        std::vector<int> pdq_data = original_ints;
        std::vector<int> exp_ints = original_ints;
        std::sort(exp_ints.begin(), exp_ints.end());
        algoat::sorting::pdqsort(pdq_data);
        if (pdq_data != exp_ints)
            __builtin_trap();
    }

    // O(N^2) algorithms (only run on small inputs to avoid fuzzer timeouts)
    if (num_elements < 500) {
        test_algo(SelectionSort{}, false);
        test_algo(BubbleSort{}, true);
        test_algo(InsertionSort{}, true);
        test_algo(GnomeSort{}, false);
        test_algo(CycleSort{}, false);
    }

    // BitonicSort (requires power of 2)
    {
        size_t n = 1;
        while (n < num_elements)
            n *= 2;

        std::vector<StableItem> bitonic_data = original;
        while (bitonic_data.size() < n)
            bitonic_data.push_back({0, -1});

        std::vector<int> bit_exp(bitonic_data.size());
        for (size_t i = 0; i < bitonic_data.size(); ++i)
            bit_exp[i] = bitonic_data[i].key;
        std::sort(bit_exp.begin(), bit_exp.end());

        BitonicSort{}.sort(std::span{bitonic_data});

        std::vector<int> bit_res(bitonic_data.size());
        for (size_t i = 0; i < bitonic_data.size(); ++i)
            bit_res[i] = bitonic_data[i].key;

        if (bit_res != bit_exp)
            __builtin_trap();
    }

    // Linear Sorts (CountingSort, PigeonholeSort, BucketSort)
    if (num_elements > 0) {
        auto [min_it, max_it] = std::minmax_element(original_ints.begin(), original_ints.end());
        long long range = static_cast<long long>(*max_it) - static_cast<long long>(*min_it);
        if (range >= 0 && range < 1000000) {
            std::vector<int> expected_ints = original_ints;
            std::sort(expected_ints.begin(), expected_ints.end());

            auto test_linear = [&](auto algo) {
                std::vector<int> test_data = original_ints;
                algo.sort(std::span{test_data});
                if (test_data != expected_ints)
                    __builtin_trap();
            };

            test_linear(CountingSort{});
            test_linear(PigeonholeSort{});
            test_linear(BucketSort{});
        }
    }

    // Radix sorts
    std::vector<int> expected_ints = original_ints;
    std::sort(expected_ints.begin(), expected_ints.end());

    auto test_radix = [&](auto algo) {
        std::vector<int> test_data = original_ints;
        algo.sort(std::span{test_data});
        if (test_data != expected_ints)
            __builtin_trap();
    };

    test_radix(RadixSortLSD{});
    test_radix(RadixSortMSD{});
    test_radix(RadixSortInPlaceMSD{});

    // Boolean sort (bool array and integer zero-partition)
    if (num_elements > 0) {
        auto bool_arr = std::make_unique<bool[]>(num_elements);
        size_t expected_true = 0;
        for (size_t i = 0; i < num_elements; ++i) {
            bool_arr[i] = (original_ints[i] % 2 != 0);
            if (bool_arr[i])
                ++expected_true;
        }
        std::span<bool> bool_span(bool_arr.get(), num_elements);
        algoat::sorting::sort_boolean(bool_span);
        if (!std::is_sorted(bool_span.begin(), bool_span.end()))
            __builtin_trap();
        size_t actual_true = 0;
        for (size_t i = 0; i < num_elements; ++i) {
            if (bool_span[i])
                ++actual_true;
        }
        if (actual_true != expected_true)
            __builtin_trap();

        // Integer zero-partition overload
        std::vector<int> int_bool_data = original_ints;
        algoat::sorting::sort_boolean(std::span<int>{int_bool_data});
        bool seen_nonzero = false;
        for (int val : int_bool_data) {
            if (val != 0) {
                seen_nonzero = true;
            } else if (seen_nonzero) {
                __builtin_trap();
            }
        }
    }

    // Compile-Time Algorithm Selector (algoat::meta::static_sort)
    {
        // 1. Default policy (unstable / introsort)
        std::vector<int> test_static_default = original_ints;
        algoat::meta::static_sort(std::span{test_static_default});
        if (test_static_default != expected_ints)
            __builtin_trap();

        // 2. Stable in-place policy (BlockSort)
        std::vector<StableItem> test_static_stable = original;
        algoat::meta::static_sort(std::span{test_static_stable}, p::stable | p::in_place);
        for (size_t i = 0; i < test_static_stable.size(); ++i) {
            if (test_static_stable[i].key != stable_expected[i].key ||
                test_static_stable[i].original_index != stable_expected[i].original_index)
                __builtin_trap();
        }

        // 3. Force Radix policy
        std::vector<int> test_static_radix = original_ints;
        algoat::meta::static_sort(std::span{test_static_radix}, p::force_radix);
        if (test_static_radix != expected_ints)
            __builtin_trap();

        // 4. Nearly sorted policy (TimSort)
        std::vector<StableItem> test_static_nearly = stable_expected;
        if (test_static_nearly.size() >= 2) {
            std::swap(test_static_nearly[0], test_static_nearly[1]);
        }
        algoat::meta::static_sort(std::span{test_static_nearly}, p::nearly_sorted);
        if (!std::is_sorted(test_static_nearly.begin(), test_static_nearly.end(),
                            [](const auto& a, const auto& b) { return a.key < b.key; }))
            __builtin_trap();

        // 5. Fixed-extent sorting networks (N = 2, 4, 8, 16)
        if (num_elements >= 2) {
            std::array<int, 2> buf;
            std::memcpy(buf.data(), original_ints.data(), 2 * sizeof(int));
            std::array<int, 2> exp_buf = buf;
            std::sort(exp_buf.begin(), exp_buf.end());
            algoat::meta::static_sort(std::span<int, 2>{buf});
            if (buf != exp_buf)
                __builtin_trap();
        }
        if (num_elements >= 4) {
            std::array<int, 4> buf;
            std::memcpy(buf.data(), original_ints.data(), 4 * sizeof(int));
            std::array<int, 4> exp_buf = buf;
            std::sort(exp_buf.begin(), exp_buf.end());
            algoat::meta::static_sort(std::span<int, 4>{buf});
            if (buf != exp_buf)
                __builtin_trap();
        }
        if (num_elements >= 8) {
            std::array<int, 8> buf;
            std::memcpy(buf.data(), original_ints.data(), 8 * sizeof(int));
            std::array<int, 8> exp_buf = buf;
            std::sort(exp_buf.begin(), exp_buf.end());
            algoat::meta::static_sort(std::span<int, 8>{buf});
            if (buf != exp_buf)
                __builtin_trap();
        }
        if (num_elements >= 16) {
            std::array<int, 16> buf;
            std::memcpy(buf.data(), original_ints.data(), 16 * sizeof(int));
            std::array<int, 16> exp_buf = buf;
            std::sort(exp_buf.begin(), exp_buf.end());
            algoat::meta::static_sort(std::span<int, 16>{buf});
            if (buf != exp_buf)
                __builtin_trap();
        }
    }

    // Spatial Curve Sorts (Morton, Hilbert, Hybrid SFC)
    if (size % sizeof(std::complex<float>) == 0 && size > 0) {
        size_t complex_num_elements = size / sizeof(std::complex<float>);
        std::vector<std::complex<float>> original_complex(complex_num_elements);
        std::memcpy(original_complex.data(), data, size);

        // 1. Hybrid SFC
        {
            std::vector<std::complex<float>> test_data = original_complex;
            algoat::numerics::sort_complex_hybrid(std::span{test_data});
            if (!std::is_sorted(test_data.begin(), test_data.end(),
                                algoat::numerics::HybridCompare{})) {
                __builtin_trap();
            }
        }

        // 2. Morton curve
        {
            std::vector<std::complex<float>> test_data = original_complex;
            algoat::numerics::sort_complex_morton(std::span{test_data});
            if (!std::is_sorted(test_data.begin(), test_data.end(),
                                algoat::numerics::MortonCompare{})) {
                __builtin_trap();
            }
        }

        // 3. Hilbert sort
        {
            std::vector<std::complex<float>> test_data = original_complex;
            HilbertSort{}.sort(std::span{test_data});
            if (!std::is_sorted(test_data.begin(), test_data.end(),
                                algoat::numerics::HilbertCompare{})) {
                __builtin_trap();
            }
        }
    }

    // Float16 non-comparative sort
    if (size % sizeof(uint16_t) == 0 && size > 0) {
        size_t f16_count = size / sizeof(uint16_t);
        std::vector<uint16_t> f16_data(f16_count);
        std::memcpy(f16_data.data(), data, size);

        std::vector<uint16_t> expected_f16 = f16_data;
        std::sort(expected_f16.begin(), expected_f16.end(), algoat::numerics::Float16Compare{});

        algoat::numerics::sort_float16(std::span<uint16_t>{f16_data});
        if (f16_data != expected_f16)
            __builtin_trap();
    }

    return 0;
}
