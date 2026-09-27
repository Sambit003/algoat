#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::sorting;

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
    if (size % sizeof(int) != 0)
        return 0;
    size_t num_elements = size / sizeof(int);

    std::vector<int> original_ints(num_elements);
    std::memcpy(original_ints.data(), data, size);

    std::vector<StableItem> original;
    for (size_t i = 0; i < num_elements; ++i) {
        original.push_back({original_ints[i], static_cast<int>(i)});
    }

    std::vector<StableItem> expected = original;
    std::sort(expected.begin(), expected.end(), [](const StableItem& a, const StableItem& b) {
        if (a.key != b.key)
            return a.key < b.key;
        return a.original_index <
               b.original_index; // std::sort is unstable, so we force deterministic output for
                                 // tests by comparing index
    });

    std::vector<StableItem> stable_expected = original;
    std::stable_sort(stable_expected.begin(), stable_expected.end(),
                     [](const StableItem& a, const StableItem& b) {
                         return a.key < b.key; // stable_sort will preserve index automatically
                     });

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
            // For unstable, we only care that the keys are correctly sorted
            // (Conservation/Monotonicity)
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

    // Fast O(N log N) algorithms
    test_algo(QuickSort{}, false);
    test_algo(HeapSort{}, false);
    test_algo(MergeSort{}, true);
    test_algo(IntroSort{}, false);
    test_algo(TimSort{}, true);
    test_algo(BlockSort{}, true);
    test_algo(ShellSort{}, false);
    test_algo(CombSort{}, false);

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

    // Linear Sorts (only run if range is small enough to avoid OOM)
    // Actually, linear sorts don't work with StableItem because they need integers!
    // We'll test linear sorts with just plain ints.
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
    auto test_radix = [&](auto algo) {
        std::vector<int> expected_ints = original_ints;
        std::sort(expected_ints.begin(), expected_ints.end());
        std::vector<int> test_data = original_ints;
        algo.sort(std::span{test_data});
        if (test_data != expected_ints)
            __builtin_trap();
    };

    test_radix(RadixSortLSD{});
    test_radix(RadixSortMSD{});
    test_radix(RadixSortInPlaceMSD{});

    return 0;
}
