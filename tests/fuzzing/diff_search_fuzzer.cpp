#include "algoat/searching/searching.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::searching;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size % sizeof(int) != 0 || size == 0)
        return 0;

    // Use the first int as the target, the rest as the array
    int target;
    std::memcpy(&target, data, sizeof(int));

    size_t num_elements = (size / sizeof(int)) - 1;
    std::vector<int> original(num_elements);
    if (num_elements > 0) {
        std::memcpy(original.data(), data + sizeof(int), num_elements * sizeof(int));
    }

    // Prepare a sorted unique array for advanced searches
    std::vector<int> sorted_data = original;
    std::sort(sorted_data.begin(), sorted_data.end());
    auto unique_it = std::unique(sorted_data.begin(), sorted_data.end());
    sorted_data.erase(unique_it, sorted_data.end());

    // Baseline (std::find) on original
    auto std_it = std::find(original.begin(), original.end(), target);
    bool expected_found = (std_it != original.end());

    // Test LinearSearch on original
    auto lin_res = LinearSearch{}.search(std::span<const int>{original}, target);
    if (lin_res.has_value() != expected_found)
        __builtin_trap();
    if (lin_res.has_value() && original[lin_res.value()] != target)
        __builtin_trap();

    // Baseline (std::binary_search) on sorted_data
    bool sorted_expected = std::binary_search(sorted_data.begin(), sorted_data.end(), target);

    auto test_sorted_algo = [&](auto algo) {
        auto res = algo.search(std::span<const int>{sorted_data}, target);
        if (res.has_value() != sorted_expected)
            __builtin_trap();
        if (res.has_value() && sorted_data[res.value()] != target)
            __builtin_trap();
    };

    test_sorted_algo(BinarySearch{});
    test_sorted_algo(InterpolationSearch{});
    test_sorted_algo(AdaptiveBinarySearch{});
    test_sorted_algo(HybridInterpolationSearch{});

    return 0;
}
