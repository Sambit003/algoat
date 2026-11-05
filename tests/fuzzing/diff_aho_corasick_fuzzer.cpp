#include "algoat/dp/aho_corasick.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    // Byte 0: number of keywords (1 to 8)
    const size_t num_keywords = (static_cast<size_t>(data[0]) % 8) + 1;
    size_t cursor = 1;

    algoat::dp::AhoCorasick<char> ac;
    std::vector<std::string> keywords;

    for (size_t i = 0; i < num_keywords && cursor < size; ++i) {
        // Next byte: length of keyword (1 to 16)
        size_t kw_len = (static_cast<size_t>(data[cursor++]) % 16) + 1;
        if (cursor + kw_len > size) {
            kw_len = size - cursor;
        }
        if (kw_len > 0) {
            std::string kw(reinterpret_cast<const char*>(data + cursor), kw_len);
            cursor += kw_len;
            keywords.push_back(kw);
            ac.add_keyword(kw);
        }
    }

    if (keywords.empty()) {
        return 0;
    }

    ac.build();

    // Remaining bytes are the search text
    std::string_view text(reinterpret_cast<const char*>(data + cursor), size - cursor);

    // 1. Run find_all (non-overlapping)
    const auto non_overlapping = ac.find_all(text);
    size_t last_end = 0;
    for (const auto& m : non_overlapping) {
        if (m.pattern_id >= keywords.size()) {
            __builtin_trap();
        }
        if (m.length != keywords[m.pattern_id].size()) {
            __builtin_trap();
        }
        if (m.end_pos > text.size() || m.end_pos < m.length) {
            __builtin_trap();
        }
        size_t start = m.end_pos - m.length;
        if (start < last_end) {
            __builtin_trap(); // Must be non-overlapping
        }
        last_end = m.end_pos;
        if (text.substr(start, m.length) != keywords[m.pattern_id]) {
            __builtin_trap(); // Must be exact match
        }
    }

    // 2. Run find_first
    const auto first = ac.find_first(text);
    if (!non_overlapping.empty()) {
        if (!first.has_value() || first->pattern_id != non_overlapping[0].pattern_id ||
            first->end_pos != non_overlapping[0].end_pos) {
            __builtin_trap();
        }
    } else {
        if (first.has_value()) {
            __builtin_trap();
        }
    }

    // 3. Run find_all_callback
    size_t callback_count = 0;
    ac.find_all_callback(text, [&](algoat::dp::AcMatch m) {
        if (m.pattern_id >= keywords.size()) {
            __builtin_trap();
        }
        if (m.length != keywords[m.pattern_id].size()) {
            __builtin_trap();
        }
        if (m.end_pos > text.size() || m.end_pos < m.length) {
            __builtin_trap();
        }
        size_t start = m.end_pos - m.length;
        if (text.substr(start, m.length) != keywords[m.pattern_id]) {
            __builtin_trap();
        }
        ++callback_count;
        return true;
    });

    if (callback_count < non_overlapping.size()) {
        __builtin_trap();
    }

    return 0;
}
