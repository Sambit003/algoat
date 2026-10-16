#include "algoat/searching/eytzinger_search_tree.hpp"
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
    test_sorted_algo(BranchlessBinarySearch{});
    test_sorted_algo(GallopingSearch{});

    // Test branchless_lower_bound and galloping_search directly against std::lower_bound
    auto std_lb = std::lower_bound(sorted_data.begin(), sorted_data.end(), target);
    auto bl_lb = branchless_lower_bound(sorted_data.begin(), sorted_data.end(), target);
    if (bl_lb != std_lb)
        __builtin_trap();
    auto gs_lb = galloping_search(sorted_data.begin(), sorted_data.end(), target);
    if (gs_lb != std_lb)
        __builtin_trap();

    // Test EytzingerSearchTree on sorted unique data
    if (!sorted_data.empty()) {
        EytzingerSearchTree<int> tree(std::span<const int>{sorted_data});
        auto lower_idx = tree.lower_bound(target);
        auto std_lb = std::lower_bound(sorted_data.begin(), sorted_data.end(), target);
        size_t expected_lb = static_cast<size_t>(std_lb - sorted_data.begin());
        if (lower_idx != expected_lb)
            __builtin_trap();
    }

    return 0;
}
