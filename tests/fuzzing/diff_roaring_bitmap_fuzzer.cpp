#include "algoat/succinct/roaring_bitmap.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <unordered_set>
#include <vector>

using namespace algoat::succinct;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0)
        return 0;

    // 1. Fuzz Deserialization with arbitrary bytes (robustness test)
    try {
        RoaringBitmap bm = RoaringBitmap::deserialize(std::span<const uint8_t>(data, size));
        // If successful, test that basic methods don't crash
        volatile uint64_t card = bm.cardinality();
        volatile bool empty = bm.empty();
        (void)card;
        (void)empty;

        // Roundtrip serialization
        auto reserialized = bm.serialize();
        RoaringBitmap bm2 = RoaringBitmap::deserialize(reserialized);
        if (bm.cardinality() != bm2.cardinality()) {
            __builtin_trap();
        }
    } catch (const std::runtime_error&) {
        // Expected on invalid/corrupted bytes
    }

    // 2. Differential Fuzzing against std::unordered_set
    if (size >= 5) {
        RoaringBitmap bm;
        std::unordered_set<uint32_t> ref;

        size_t offset = 0;
        while (offset + 5 <= size) {
            uint8_t op = data[offset++];
            uint32_t val = 0;
            std::memcpy(&val, data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);

            switch (op % 4) {
            case 0: { // add
                bool bm_res = bm.add(val);
                bool ref_res = ref.insert(val).second;
                if (bm_res != ref_res)
                    __builtin_trap();
                break;
            }
            case 1: { // remove
                bool bm_res = bm.remove(val);
                bool ref_res = (ref.erase(val) > 0);
                if (bm_res != ref_res)
                    __builtin_trap();
                break;
            }
            case 2: { // contains
                bool bm_res = bm.contains(val);
                bool ref_res = (ref.find(val) != ref.end());
                if (bm_res != ref_res)
                    __builtin_trap();
                break;
            }
            case 3: { // run_optimize
                bm.run_optimize();
                if (bm.cardinality() != ref.size())
                    __builtin_trap();
                break;
            }
            }
        }

        if (bm.cardinality() != ref.size()) {
            __builtin_trap();
        }
    }

    return 0;
}
