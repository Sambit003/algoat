#include "algoat/succinct/elias_fano.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

using namespace algoat::succinct;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) {
        return 0;
    }

    uint32_t raw_n = 0;
    std::memcpy(&raw_n, data, sizeof(uint32_t));
    size_t n = raw_n % 1024; // Limit to 1024 elements for fuzz loop speed

    uint32_t query_raw = 0;
    std::memcpy(&query_raw, data + 4, sizeof(uint32_t));

    size_t byte_offset = 8;
    std::vector<uint64_t> seq;
    seq.reserve(n);

    for (size_t i = 0; i < n && (byte_offset + sizeof(uint32_t) <= size); ++i) {
        uint32_t val = 0;
        std::memcpy(&val, data + byte_offset, sizeof(uint32_t));
        seq.push_back(static_cast<uint64_t>(val));
        byte_offset += sizeof(uint32_t);
    }

    std::sort(seq.begin(), seq.end());
    uint64_t universe = seq.empty() ? 100ULL : (seq.back() + 50ULL);

    EliasFanoEncoder encoder(seq.size(), universe);
    for (uint64_t x : seq) {
        encoder.push(x);
    }
    std::vector<uint8_t> buffer = encoder.finalize();

    EliasFanoList list(buffer);
    if (list.size() != seq.size() || list.universe() != universe) {
        __builtin_trap();
    }

    for (size_t i = 0; i < seq.size(); ++i) {
        if (list.at(i) != seq[i]) {
            __builtin_trap();
        }
    }

    // Iterator check
    size_t iter_count = 0;
    for (uint64_t val : list) {
        if (val != seq[iter_count++]) {
            __builtin_trap();
        }
    }
    if (iter_count != seq.size()) {
        __builtin_trap();
    }

    // Successor query check
    uint64_t target = static_cast<uint64_t>(query_raw % (universe + 10ULL));
    size_t pos = list.lower_bound_pos(target);
    uint64_t next_val = list.next_geq(target);

    auto std_it = std::lower_bound(seq.begin(), seq.end(), target);
    size_t expected_pos = static_cast<size_t>(std::distance(seq.begin(), std_it));
    uint64_t expected_val = (std_it != seq.end()) ? *std_it : universe;

    if (pos != expected_pos || next_val != expected_val) {
        __builtin_trap();
    }

    // Zero-copy span check
    EliasFanoList span_list(std::span<const uint8_t>{buffer.data(), buffer.size()});
    if (span_list.size() != list.size() || span_list.universe() != list.universe()) {
        __builtin_trap();
    }

    return 0;
}
