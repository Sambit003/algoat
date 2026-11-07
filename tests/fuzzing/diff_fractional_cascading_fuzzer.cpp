/**
 * @file diff_fractional_cascading_fuzzer.cpp
 * @brief Differential fuzzer for FractionalCascading against std::lower_bound.
 */

#include "algoat/searching/fractional_cascading.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::searching;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < sizeof(int) + 1) {
        return 0;
    }

    int target = 0;
    std::memcpy(&target, data, sizeof(int));

    uint8_t raw_k = data[sizeof(int)];
    std::size_t k = (raw_k % 16) + 1; // 1 to 16 lists

    const uint8_t* payload = data + sizeof(int) + 1;
    size_t payload_bytes = size - (sizeof(int) + 1);
    size_t num_ints = payload_bytes / sizeof(int);

    std::vector<int> all_ints(num_ints);
    if (num_ints > 0) {
        std::memcpy(all_ints.data(), payload, num_ints * sizeof(int));
    }

    std::vector<std::vector<int>> lists(k);
    for (size_t i = 0; i < num_ints; ++i) {
        lists[i % k].push_back(all_ints[i]);
    }

    for (size_t i = 0; i < k; ++i) {
        std::sort(lists[i].begin(), lists[i].end());
    }

    FractionalCascading<int> fc(lists);
    if (fc.list_count() != k) {
        __builtin_trap();
    }

    auto results = fc.search_all(target);
    if (results.size() != k) {
        __builtin_trap();
    }

    std::vector<std::size_t> span_results(k);
    fc.search_all_to_span(target, span_results);
    if (span_results != results) {
        __builtin_trap();
    }

    for (size_t i = 0; i < k; ++i) {
        auto expected_it = std::lower_bound(lists[i].begin(), lists[i].end(), target);
        std::size_t expected_idx = static_cast<std::size_t>(expected_it - lists[i].begin());
        if (results[i] != expected_idx) {
            __builtin_trap();
        }
    }

    return 0;
}
