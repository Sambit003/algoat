#include "algoat/succinct/fm_index.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using algoat::succinct::FMIndex;
using algoat::succinct::FMIndexConfig;

namespace {

size_t naive_count(std::string_view text, std::string_view pattern) {
    if (pattern.empty() || text.empty()) {
        return 0;
    }
    size_t count = 0;
    size_t pos = text.find(pattern, 0);
    while (pos != std::string_view::npos) {
        count++;
        pos = text.find(pattern, pos + 1);
    }
    return count;
}

std::vector<size_t> naive_locate(std::string_view text, std::string_view pattern) {
    std::vector<size_t> positions;
    if (pattern.empty() || text.empty()) {
        return positions;
    }
    size_t pos = text.find(pattern, 0);
    while (pos != std::string_view::npos) {
        positions.push_back(pos);
        pos = text.find(pattern, pos + 1);
    }
    return positions;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 12) {
        return 0;
    }

    uint32_t raw_n = 0;
    std::memcpy(&raw_n, data, sizeof(uint32_t));
    size_t n = raw_n % 256; // Cap to 256 for fast fuzzer turnaround

    uint32_t query_raw1 = 0;
    std::memcpy(&query_raw1, data + 4, sizeof(uint32_t));

    uint32_t query_raw2 = 0;
    std::memcpy(&query_raw2, data + 8, sizeof(uint32_t));

    size_t byte_offset = 12;
    std::string text;
    text.reserve(n);
    for (size_t i = 0; i < n && byte_offset < size; ++i) {
        text.push_back(static_cast<char>(data[byte_offset++]));
    }
    n = text.size();

    FMIndexConfig cfg;
    cfg.sa_sample_rate = 1 + (query_raw1 % 16);
    cfg.build_isa_samples = ((query_raw1 >> 4) & 1);

    FMIndex idx{text, cfg};
    if (idx.text_length() != n) {
        __builtin_trap();
    }

    if (n > 0) {
        // Query 1: Existing substring
        size_t p_start = (query_raw1 >> 8) % n;
        size_t p_len = 1 + ((query_raw1 >> 16) % std::min<size_t>(8, n - p_start));
        std::string pat1 = text.substr(p_start, p_len);

        size_t exp_count1 = naive_count(text, pat1);
        if (idx.count(pat1) != exp_count1) {
            __builtin_trap();
        }

        auto exp_locs1 = naive_locate(text, pat1);
        auto act_locs1 = idx.locate(pat1);
        std::sort(act_locs1.begin(), act_locs1.end());
        if (act_locs1 != exp_locs1) {
            __builtin_trap();
        }

        // Query 2: Extract
        size_t ext_pos = query_raw2 % n;
        size_t ext_len = (query_raw2 >> 8) % (n - ext_pos + 1);
        std::string extracted = idx.extract(ext_pos, ext_len);
        if (extracted != text.substr(ext_pos, ext_len)) {
            __builtin_trap();
        }

        // Query 3: Serialization round-trip
        std::vector<uint8_t> serialized = idx.serialize();
        FMIndex restored = FMIndex::deserialize(serialized);
        if (restored.text_length() != idx.text_length() || restored.count(pat1) != exp_count1 ||
            restored.extract(ext_pos, ext_len) != extracted) {
            __builtin_trap();
        }
    }

    return 0;
}
