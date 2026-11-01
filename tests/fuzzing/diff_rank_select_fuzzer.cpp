#include "algoat/succinct/rank_select.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::succinct;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    // Determine number of bits (0 to 8192)
    uint32_t raw_n = 0;
    std::memcpy(&raw_n, data, sizeof(uint32_t));
    size_t num_bits = raw_n % 8193;

    size_t byte_offset = 4;
    size_t payload_bytes = (num_bits + 7) / 8;

    std::vector<bool> ref_bits(num_bits, false);
    for (size_t i = 0; i < num_bits; ++i) {
        size_t b_idx = byte_offset + (i / 8);
        if (b_idx < size) {
            ref_bits[i] = ((data[b_idx] >> (i % 8)) & 1U) != 0;
        }
    }

    RankSelectBitVector bv(ref_bits);

    if (bv.num_bits() != num_bits) {
        __builtin_trap();
    }

    uint64_t ref_ones = 0;
    for (size_t i = 0; i < num_bits; ++i) {
        if (ref_bits[i]) {
            ++ref_ones;
        }
    }

    if (bv.total_ones() != ref_ones) {
        __builtin_trap();
    }

    // Rank / access verification
    size_t cur_ones = 0;
    for (size_t i = 0; i <= num_bits; ++i) {
        uint64_t r1 = bv.rank1(i);
        uint64_t r0 = bv.rank0(i);
        if (r1 != cur_ones) {
            __builtin_trap();
        }
        if (r0 != (i - cur_ones)) {
            __builtin_trap();
        }
        if (i < num_bits) {
            bool acc = bv.access(i);
            if (acc != ref_bits[i]) {
                __builtin_trap();
            }
            if (ref_bits[i]) {
                ++cur_ones;
            }
        }
    }

    // Select verification
    size_t query_offset = byte_offset + payload_bytes;
    while (query_offset + sizeof(uint32_t) <= size) {
        uint32_t val = 0;
        std::memcpy(&val, data + query_offset, sizeof(uint32_t));
        query_offset += sizeof(uint32_t);

        if (ref_ones > 0) {
            uint64_t k1 = 1 + (val % ref_ones);
            size_t p1 = bv.select1(k1);
            if (p1 >= num_bits || !bv.access(p1) || bv.rank1(p1) != (k1 - 1)) {
                __builtin_trap();
            }
        }

        uint64_t ref_zeros = num_bits - ref_ones;
        if (ref_zeros > 0) {
            uint64_t k0 = 1 + (val % ref_zeros);
            size_t p0 = bv.select0(k0);
            if (p0 >= num_bits || bv.access(p0) || bv.rank0(p0) != (k0 - 1)) {
                __builtin_trap();
            }
        }
    }

    return 0;
}
