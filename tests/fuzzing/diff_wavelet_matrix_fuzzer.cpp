#include "algoat/succinct/wavelet_matrix.hpp"

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
    size_t n = raw_n % 512; // Cap to 512 for fast differential iteration

    uint32_t query_raw = 0;
    std::memcpy(&query_raw, data + 4, sizeof(uint32_t));

    size_t byte_offset = 8;
    std::vector<uint8_t> seq;
    seq.reserve(n);

    for (size_t i = 0; i < n && (byte_offset < size); ++i) {
        seq.push_back(data[byte_offset++]);
    }
    n = seq.size();

    WaveletMatrix<uint8_t> wm(seq);

    if (wm.size() != n) {
        __builtin_trap();
    }

    // 1. Access differential check
    for (size_t i = 0; i < n; ++i) {
        if (wm.access(i) != seq[i]) {
            __builtin_trap();
        }
    }

    // 2. Rank and Select differential check on sampled characters
    uint8_t test_char = static_cast<uint8_t>(query_raw & 0xFF);
    size_t test_pref = (n == 0) ? 0 : ((query_raw >> 8) % (n + 1));

    size_t expected_rank = 0;
    for (size_t i = 0; i < test_pref; ++i) {
        if (seq[i] == test_char) {
            ++expected_rank;
        }
    }

    if (wm.rank(test_char, test_pref) != expected_rank) {
        __builtin_trap();
    }

    // Select check
    size_t total_c = 0;
    std::vector<size_t> pos_c;
    for (size_t i = 0; i < n; ++i) {
        if (seq[i] == test_char) {
            pos_c.push_back(i);
            ++total_c;
        }
    }

    if (total_c > 0) {
        size_t test_k = ((query_raw >> 16) % total_c) + 1;
        size_t p = wm.select(test_char, test_k);
        if (p != pos_c[test_k - 1] || wm.access(p) != test_char) {
            __builtin_trap();
        }
    }

    // 3. Range Quantile & Range Count differential check
    if (n >= 2) {
        size_t l = (query_raw >> 8) % n;
        size_t r = ((query_raw >> 16) % n) + 1;
        if (l >= r) {
            std::swap(l, r);
            if (l == r) {
                if (r < n)
                    ++r;
                else if (l > 0)
                    --l;
            }
        }

        if (l < r) {
            size_t k = (query_raw >> 24) % (r - l);
            std::vector<uint8_t> sub(seq.begin() + l, seq.begin() + r);
            std::nth_element(sub.begin(), sub.begin() + k, sub.end());
            if (wm.range_quantile(l, r, k) != sub[k]) {
                __builtin_trap();
            }

            uint8_t lo = static_cast<uint8_t>(query_raw & 0x7F);
            uint8_t hi = static_cast<uint8_t>((query_raw >> 7) & 0xFF);
            if (lo > hi)
                std::swap(lo, hi);

            size_t expected_count = 0;
            for (size_t i = l; i < r; ++i) {
                if (seq[i] >= lo && seq[i] <= hi) {
                    ++expected_count;
                }
            }
            if (wm.range_count(l, r, lo, hi) != expected_count) {
                __builtin_trap();
            }
        }
    }

    return 0;
}
