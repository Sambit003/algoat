#include "algoat/sketches/hyperloglog.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace algoat::sketches;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 4) {
        return 0;
    }

    // 1. Fuzz Deserialization with arbitrary unconstrained inputs
    try {
        auto hll = HyperLogLog<14>::deserialize(std::span<const uint8_t>(data, size));
        volatile uint64_t count = hll.count();
        (void)count;

        // Roundtrip serialization check on parsed valid object
        auto reserialized = hll.serialize();
        auto hll2 = HyperLogLog<14>::deserialize(reserialized);
        (void)hll2.count();
    } catch (const std::exception&) {
        // Expected on corrupted byte streams
    }

    // 2. Fuzz HLL additions with raw byte sequences
    HyperLogLog<12> dynamic_hll;
    const std::size_t chunk_size = 8;
    for (std::size_t i = 0; i + chunk_size <= size; i += chunk_size) {
        uint64_t val = 0;
        std::memcpy(&val, data + i, sizeof(uint64_t));
        dynamic_hll.add_hash(val);
    }

    volatile uint64_t dynamic_count = dynamic_hll.count();
    (void)dynamic_count;

    return 0;
}
