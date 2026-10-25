#include "algoat/strings/substring_search.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return 0;
    }

    // 1. Differential test on arbitrary fuzzer inputs
    const size_t needle_len = static_cast<size_t>(data[0]) % size;
    const size_t haystack_len = size - 1 - needle_len;

    const char* needle_ptr = reinterpret_cast<const char*>(data + 1);
    const char* haystack_ptr = reinterpret_cast<const char*>(data + 1 + needle_len);

    std::string_view needle(needle_ptr, needle_len);
    std::string_view haystack(haystack_ptr, haystack_len);

    const auto expected = haystack.find(needle);

    const auto actual_simd = algoat::strings::find_substring(haystack, needle);
    if (actual_simd != expected) {
        __builtin_trap();
    }

    const auto actual_twoway = algoat::strings::detail::two_way_search(haystack, needle);
    if (actual_twoway != expected) {
        __builtin_trap();
    }

    // 2. Differential test on pathological repetitive patterns (a^N + b vs a^M + b)
    if (size >= 8) {
        const size_t repeat_count = (size % 1000) + 50;
        const size_t pat_len = (needle_len % (repeat_count / 2)) + 1;
        const char rep_char = static_cast<char>(data[1]);
        const char diff_char = static_cast<char>(data[2]);

        std::string path_haystack(repeat_count, rep_char);
        path_haystack.push_back(diff_char);

        std::string path_needle(pat_len, rep_char);
        path_needle.push_back(diff_char);

        std::string_view path_hs(path_haystack);
        std::string_view path_ne(path_needle);

        const auto path_exp = path_hs.find(path_ne);
        const auto path_act_simd = algoat::strings::find_substring(path_hs, path_ne);
        if (path_act_simd != path_exp) {
            __builtin_trap();
        }

        const auto path_act_twoway = algoat::strings::detail::two_way_search(path_hs, path_ne);
        if (path_act_twoway != path_exp) {
            __builtin_trap();
        }
    }

    // 3. Test find_all_substrings
    const auto all_matches = algoat::strings::find_all_substrings(haystack, needle);
    if (needle.empty()) {
        if (!all_matches.empty()) {
            __builtin_trap();
        }
    } else {
        std::vector<std::string_view::size_type> naive_matches;
        for (std::size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
            if (haystack.substr(i, needle.size()) == needle) {
                naive_matches.push_back(i);
            }
        }
        if (all_matches != naive_matches) {
            __builtin_trap();
        }
    }

    return 0;
}
