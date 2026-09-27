#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/sorting.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::sorting;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size % sizeof(int) != 0) {
        return 0; // ensure aligned for int
    }

    size_t num_elements = size / sizeof(int);
    std::vector<int> original(num_elements);
    std::memcpy(original.data(), data, size);

    // Baseline using std::sort
    std::vector<int> expected = original;
    std::sort(expected.begin(), expected.end());

    // Test QuickSort
    std::vector<int> quick_data = original;
    QuickSort{}.sort(std::span{quick_data});
    if (quick_data != expected) {
        __builtin_trap();
    }

    // Test HeapSort
    std::vector<int> heap_data = original;
    HeapSort{}.sort(std::span{heap_data});
    if (heap_data != expected) {
        __builtin_trap();
    }

    // Test MergeSort against std::stable_sort
    std::vector<int> stable_expected = original;
    std::stable_sort(stable_expected.begin(), stable_expected.end());

    std::vector<int> merge_data = original;
    MergeSort{}.sort(std::span{merge_data});
    if (merge_data != stable_expected) {
        // even though int is identical, testing equality of result
        // if this was object it would matter more, but for int it's still good.
        if (merge_data != stable_expected) {
            __builtin_trap();
        }
    }

    return 0;
}
