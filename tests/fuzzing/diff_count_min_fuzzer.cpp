#include "algoat/sketches/count_min.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

using namespace algoat::sketches;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 8) {
        return 0;
    }

    CountMinSketch<uint64_t, uint32_t> cms_cu(0.01, 0.05, true);
    CountMinSketch<uint64_t, uint32_t> cms_std(0.01, 0.05, false);
    SpaceSavingHeavyHitters<uint64_t, 32> ss;

    std::unordered_map<uint64_t, uint32_t> exact;
    uint64_t total_weight = 0;

    const std::size_t chunk_size = sizeof(uint64_t);
    for (std::size_t i = 0; i + chunk_size <= size; i += chunk_size) {
        uint64_t key = 0;
        std::memcpy(&key, data + i, sizeof(uint64_t));

        // Use low byte as count weight [1, 8]
        uint32_t count = static_cast<uint32_t>((key & 0x07) + 1);

        cms_cu.add(key, count);
        cms_std.add(key, count);
        ss.add(key, count);

        exact[key] += count;
        total_weight += count;
    }

    // Invariant verifications
    for (const auto& [key, true_f] : exact) {
        uint32_t est_cu = cms_cu.estimate(key);
        uint32_t est_std = cms_std.estimate(key);

        if (est_cu < true_f || est_std < true_f) {
            __builtin_trap();
        }
    }

    auto ss_items = ss.top_k();
    uint64_t ss_sum = 0;
    for (const auto& item : ss_items) {
        ss_sum += item.count;
        uint32_t true_f = exact[item.key];
        if (item.count < true_f || (item.count - item.error) > true_f) {
            __builtin_trap();
        }
    }

    if (ss_sum != total_weight) {
        __builtin_trap();
    }

    return 0;
}
