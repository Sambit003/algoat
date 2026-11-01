#include "algoat/sketches/bloom_cuckoo_filter.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::sketches;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 8) {
        return 0;
    }

    // 1. Fuzz BlockedBloomFilter
    BlockedBloomFilter<uint64_t> bbf(100, 0.01);
    const std::size_t chunk_size = sizeof(uint64_t);
    std::vector<uint64_t> inserted;

    for (std::size_t i = 0; i + chunk_size <= size; i += chunk_size) {
        uint64_t key = 0;
        std::memcpy(&key, data + i, sizeof(uint64_t));
        bbf.insert(key);
        inserted.push_back(key);
    }

    for (uint64_t k : inserted) {
        if (!bbf.contains(k)) {
            __builtin_trap(); // Invariant: Zero false negatives
        }
    }

    if (!inserted.empty()) {
        std::unique_ptr<bool[]> dynamic_bools = std::make_unique<bool[]>(inserted.size());
        std::span<bool> dynamic_span(dynamic_bools.get(), inserted.size());
        bbf.contains_batch(inserted, dynamic_span);
        for (std::size_t i = 0; i < inserted.size(); ++i) {
            if (!dynamic_span[i]) {
                __builtin_trap();
            }
        }
    }

    // 2. Fuzz CuckooFilter (8-bit and 16-bit variants)
    CuckooFilter<uint64_t, 8, 4> cf8(256);
    CuckooFilter<uint64_t, 16, 4> cf16(256);

    for (uint64_t k : inserted) {
        (void)cf8.insert(k);
        (void)cf16.insert(k);
    }

    for (uint64_t k : inserted) {
        (void)cf8.contains(k);
        (void)cf16.contains(k);
        (void)cf8.remove(k);
        (void)cf16.remove(k);
    }

    return 0;
}
