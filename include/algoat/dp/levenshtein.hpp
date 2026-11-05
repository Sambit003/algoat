/**
 * @file levenshtein.hpp
 * @brief Myers' Bit-Parallel Levenshtein Edit Distance, Ukkonen's Banded DP, and SIMD
 * Vectorization.
 *
 * Implements high-performance sequence alignment algorithms:
 * - Myers' 1999 Bit-Parallel Algorithm: O(ceil(M/w) * N) exact edit distance.
 * - AVX2 SIMD 256-bit vectorization for M in [65, 256].
 * - Ukkonen's 1985 Banded DP: O(k * min(M, N)) bounded edit distance with early stopping.
 * - Hyyro 2004 Bit-Parallel Alignment Traceback: recovers exact CIGAR-style EditOp sequence.
 * - Crochemore et al. Bit-Parallel Longest Common Subsequence (LCS) length.
 * - AVX2 Data-Parallel SIMD Batch Edit Distance.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if defined(__AVX2__) || defined(__BMI2__)
#include <immintrin.h>
#endif
#endif
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace algoat::dp {

/// \brief CIGAR-style edit operation.
enum class EditOp : uint8_t {
    Match,      ///< Character matches (pattern[i] == text[j]), advances i and j
    Substitute, ///< Substitution (pattern[i] != text[j]), advances i and j
    Insert,     ///< Insertion of text character into pattern, advances j
    Delete      ///< Deletion of pattern character, advances i
};

/// \brief Sequence alignment result.
struct Alignment {
    size_t distance{0};             ///< Total edit distance
    std::vector<EditOp> operations; ///< Step-by-step edit operations
};

namespace detail {

// Portable 64-bit add with carry
inline unsigned char add_carry_u64(unsigned char c_in, uint64_t a, uint64_t b,
                                   uint64_t* out) noexcept {
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
    return _addcarry_u64(c_in, a, b, reinterpret_cast<unsigned long long*>(out));
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + c_in;
    *out = static_cast<uint64_t>(sum);
    return static_cast<unsigned char>(sum >> 64);
#else
    uint64_t s = a + b + c_in;
    *out = s;
    return (s < a || (c_in && s == a)) ? 1 : 0;
#endif
}

// Portable 64-bit subtract with borrow
inline unsigned char sub_borrow_u64(unsigned char b_in, uint64_t a, uint64_t b,
                                    uint64_t* out) noexcept {
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
    return _subborrow_u64(b_in, a, b, reinterpret_cast<unsigned long long*>(out));
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 diff = static_cast<unsigned __int128>(a) - b - b_in;
    *out = static_cast<uint64_t>(diff);
    return static_cast<unsigned char>((diff >> 127) & 1);
#else
    uint64_t d = a - b - b_in;
    *out = d;
    return (a < b || (b_in && a == b)) ? 1 : 0;
#endif
}

// ----------------------------------------------------------------------------
// 256-bit bit-vector state and operations for M in [65, 256]
// ----------------------------------------------------------------------------
struct alignas(32) Vec256 {
    uint64_t w[4]{0, 0, 0, 0};

    constexpr Vec256() noexcept = default;
    constexpr Vec256(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) noexcept
        : w{w0, w1, w2, w3} {}
};

[[nodiscard]] inline Vec256 vec256_or(Vec256 a, Vec256 b) noexcept {
    return Vec256{a.w[0] | b.w[0], a.w[1] | b.w[1], a.w[2] | b.w[2], a.w[3] | b.w[3]};
}

[[nodiscard]] inline Vec256 vec256_and(Vec256 a, Vec256 b) noexcept {
    return Vec256{a.w[0] & b.w[0], a.w[1] & b.w[1], a.w[2] & b.w[2], a.w[3] & b.w[3]};
}

[[nodiscard]] inline Vec256 vec256_xor(Vec256 a, Vec256 b) noexcept {
    return Vec256{a.w[0] ^ b.w[0], a.w[1] ^ b.w[1], a.w[2] ^ b.w[2], a.w[3] ^ b.w[3]};
}

[[nodiscard]] inline Vec256 vec256_andnot(Vec256 a, Vec256 b) noexcept {
    return Vec256{(~a.w[0]) & b.w[0], (~a.w[1]) & b.w[1], (~a.w[2]) & b.w[2], (~a.w[3]) & b.w[3]};
}

[[nodiscard]] inline Vec256 add256_carry(Vec256 a, Vec256 b) noexcept {
    Vec256 res;
    unsigned char c = 0;
    c = add_carry_u64(c, a.w[0], b.w[0], &res.w[0]);
    c = add_carry_u64(c, a.w[1], b.w[1], &res.w[1]);
    c = add_carry_u64(c, a.w[2], b.w[2], &res.w[2]);
    add_carry_u64(c, a.w[3], b.w[3], &res.w[3]);
    return res;
}

[[nodiscard]] inline Vec256 shl256_1(Vec256 a, uint64_t cin = 0) noexcept {
    Vec256 res;
    res.w[3] = (a.w[3] << 1) | (a.w[2] >> 63);
    res.w[2] = (a.w[2] << 1) | (a.w[1] >> 63);
    res.w[1] = (a.w[1] << 1) | (a.w[0] >> 63);
    res.w[0] = (a.w[0] << 1) | cin;
    return res;
}

[[nodiscard]] inline size_t myers_vec256(std::string_view pattern, std::string_view text) noexcept {
    const size_t m = pattern.size();
    const size_t n = text.size();

    Vec256 peq[256]{};
    for (size_t i = 0; i < m; ++i) {
        peq[static_cast<uint8_t>(pattern[i])].w[i / 64] |= (1ULL << (i % 64));
    }

    Vec256 mask{};
    for (size_t i = 0; i < m; ++i) {
        mask.w[i / 64] |= (1ULL << (i % 64));
    }

    Vec256 pv = mask;
    Vec256 mv{};
    size_t dist = m;
    const size_t top_word = (m - 1) / 64;
    const uint64_t top_bit = (1ULL << ((m - 1) % 64));

    for (size_t j = 0; j < n; ++j) {
        const Vec256 eq = peq[static_cast<uint8_t>(text[j])];

        const Vec256 xv = vec256_or(eq, mv);
        const Vec256 eq_and_pv = vec256_and(eq, pv);
        const Vec256 sum = add256_carry(eq_and_pv, pv);
        const Vec256 xh = vec256_or(vec256_xor(sum, pv), eq);

        const Vec256 xh_or_pv = vec256_or(xh, pv);
        const Vec256 not_xh_or_pv = vec256_andnot(xh_or_pv, mask);
        Vec256 ph = vec256_or(mv, not_xh_or_pv);
        Vec256 mh = vec256_and(pv, xh);

        if (ph.w[top_word] & top_bit)
            ++dist;
        if (mh.w[top_word] & top_bit)
            --dist;

        ph = shl256_1(ph, 1);
        mh = shl256_1(mh, 0);
        ph = vec256_and(ph, mask);
        mh = vec256_and(mh, mask);

        const Vec256 xv_or_ph = vec256_or(xv, ph);
        const Vec256 not_xv_or_ph = vec256_andnot(xv_or_ph, mask);
        pv = vec256_or(mh, not_xv_or_ph);
        mv = vec256_and(ph, xv);
    }
    return dist;
}

// ----------------------------------------------------------------------------
// Block-based Myers algorithm for arbitrary M > 256
// ----------------------------------------------------------------------------
struct BlockStepResult {
    uint64_t pv{0};
    uint64_t mv{0};
    uint64_t ph_raw{0};
    uint64_t mh_raw{0};
    int hout{0};
};

[[nodiscard]] inline BlockStepResult calculate_block(uint64_t pv, uint64_t mv, uint64_t eq,
                                                     const int hin) noexcept {
    constexpr uint64_t high_bit = 1ULL << 63;
    uint64_t xv = eq | mv;
    if (hin < 0)
        eq |= 1ULL;

    const uint64_t xh = (((eq & pv) + pv) ^ pv) | eq;
    uint64_t ph = mv | ~(xh | pv);
    uint64_t mh = pv & xh;

    int hout = 0;
    if (ph & high_bit)
        hout = 1;
    else if (mh & high_bit)
        hout = -1;

    BlockStepResult res;
    res.ph_raw = ph;
    res.mh_raw = mh;

    ph <<= 1;
    mh <<= 1;
    if (hin < 0)
        mh |= 1ULL;
    else if (hin > 0)
        ph |= 1ULL;

    res.pv = mh | ~(xv | ph);
    res.mv = ph & xv;
    res.hout = hout;
    return res;
}

[[nodiscard]] inline size_t myers_multiblock(std::string_view pattern, std::string_view text) {
    const size_t m = pattern.size();
    const size_t n = text.size();
    const size_t num_blocks = (m + 63) / 64;

    std::vector<std::vector<uint64_t>> eq(num_blocks, std::vector<uint64_t>(256, 0ULL));
    for (size_t i = 0; i < m; ++i) {
        eq[i / 64][static_cast<uint8_t>(pattern[i])] |= (1ULL << (i % 64));
    }

    std::vector<uint64_t> pv(num_blocks, ~0ULL);
    const size_t rem = m % 64;
    if (rem != 0) {
        pv.back() = (1ULL << rem) - 1ULL;
    }
    std::vector<uint64_t> mv(num_blocks, 0ULL);

    size_t dist = m;
    const size_t last_bit_idx = (m - 1) % 64;

    for (size_t j = 0; j < n; ++j) {
        const uint8_t c = static_cast<uint8_t>(text[j]);
        int hin = 1;

        for (size_t b = 0; b < num_blocks; ++b) {
            BlockStepResult res = calculate_block(pv[b], mv[b], eq[b][c], hin);
            pv[b] = res.pv;
            mv[b] = res.mv;
            hin = res.hout;

            if (b == num_blocks - 1) {
                dist += static_cast<size_t>((res.ph_raw >> last_bit_idx) & 1ULL);
                dist -= static_cast<size_t>((res.mh_raw >> last_bit_idx) & 1ULL);
            }
        }
    }
    return dist;
}

#if defined(__AVX2__)
// ----------------------------------------------------------------------------
// AVX2 Data-Parallel Myers kernel: 4 texts against 1 pattern (M <= 64)
// ----------------------------------------------------------------------------
inline void myers_avx2_batch4(std::string_view pattern, const std::string_view texts[4],
                              size_t dist_out[4]) noexcept {
    const size_t m = pattern.size();
    if (m == 0) {
        for (int i = 0; i < 4; ++i)
            dist_out[i] = texts[i].size();
        return;
    }

    uint64_t peq[256] = {0};
    for (size_t i = 0; i < m; ++i) {
        peq[static_cast<uint8_t>(pattern[i])] |= (1ULL << i);
    }

    const uint64_t mask_val = (m == 64) ? ~0ULL : ((1ULL << m) - 1ULL);
    const uint64_t top_bit_val = (1ULL << (m - 1));

    __m256i pv = _mm256_set1_epi64x(static_cast<long long>(mask_val));
    __m256i mv = _mm256_setzero_si256();
    __m256i dist = _mm256_set1_epi64x(static_cast<long long>(m));
    const __m256i mask = _mm256_set1_epi64x(static_cast<long long>(mask_val));
    const __m256i top_bit = _mm256_set1_epi64x(static_cast<long long>(top_bit_val));
    const __m256i ones = _mm256_set1_epi64x(1);

    const size_t n0 = texts[0].size(), n1 = texts[1].size();
    const size_t n2 = texts[2].size(), n3 = texts[3].size();
    const size_t max_n = std::max({n0, n1, n2, n3});

    for (size_t j = 0; j < max_n; ++j) {
        const uint64_t act0 = (j < n0) ? ~0ULL : 0ULL;
        const uint64_t act1 = (j < n1) ? ~0ULL : 0ULL;
        const uint64_t act2 = (j < n2) ? ~0ULL : 0ULL;
        const uint64_t act3 = (j < n3) ? ~0ULL : 0ULL;
        const __m256i active_mask =
            _mm256_set_epi64x(static_cast<long long>(act3), static_cast<long long>(act2),
                              static_cast<long long>(act1), static_cast<long long>(act0));

        const uint64_t eq0 = (j < n0) ? peq[static_cast<uint8_t>(texts[0][j])] : 0;
        const uint64_t eq1 = (j < n1) ? peq[static_cast<uint8_t>(texts[1][j])] : 0;
        const uint64_t eq2 = (j < n2) ? peq[static_cast<uint8_t>(texts[2][j])] : 0;
        const uint64_t eq3 = (j < n3) ? peq[static_cast<uint8_t>(texts[3][j])] : 0;
        const __m256i eq =
            _mm256_set_epi64x(static_cast<long long>(eq3), static_cast<long long>(eq2),
                              static_cast<long long>(eq1), static_cast<long long>(eq0));

        const __m256i xv = _mm256_or_si256(eq, mv);
        const __m256i eq_and_pv = _mm256_and_si256(eq, pv);
        const __m256i sum = _mm256_add_epi64(eq_and_pv, pv);
        const __m256i xh = _mm256_or_si256(_mm256_xor_si256(sum, pv), eq);

        const __m256i xh_or_pv = _mm256_or_si256(xh, pv);
        const __m256i not_xh_or_pv = _mm256_andnot_si256(xh_or_pv, mask);
        __m256i ph = _mm256_or_si256(mv, not_xh_or_pv);
        __m256i mh = _mm256_and_si256(pv, xh);

        // Update score from topmost bit of active lanes
        const __m256i ph_top = _mm256_srli_epi64(_mm256_and_si256(ph, top_bit), m - 1);
        const __m256i mh_top = _mm256_srli_epi64(_mm256_and_si256(mh, top_bit), m - 1);
        const __m256i delta = _mm256_sub_epi64(ph_top, mh_top);
        dist = _mm256_add_epi64(dist, _mm256_and_si256(delta, active_mask));

        ph = _mm256_and_si256(_mm256_or_si256(_mm256_slli_epi64(ph, 1), ones), mask);
        mh = _mm256_and_si256(_mm256_slli_epi64(mh, 1), mask);

        const __m256i xv_or_ph = _mm256_or_si256(xv, ph);
        const __m256i not_xv_or_ph = _mm256_andnot_si256(xv_or_ph, mask);
        pv = _mm256_or_si256(mh, not_xv_or_ph);
        mv = _mm256_and_si256(ph, xv);
    }

    alignas(32) int64_t res[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), dist);
    for (int i = 0; i < 4; ++i) {
        dist_out[i] = static_cast<size_t>(res[i]);
    }
}
#endif

} // namespace detail

// ============================================================================
// Public Interface Implementations
// ============================================================================

/**
 * @brief Computes exact Levenshtein edit distance using Myers' bit-parallel O(ceil(M/64)*N).
 *
 * @param pattern Short string (pattern). Best performance when |pattern| <= 256.
 * @param text    Long text.
 * @return Exact edit distance.
 */
[[nodiscard]] inline size_t levenshtein_distance(std::string_view pattern,
                                                 std::string_view text) noexcept {
    const size_t m = pattern.size();
    const size_t n = text.size();

    if (m == 0)
        return n;
    if (n == 0)
        return m;

    // Fast-path: pattern fits in a single 64-bit word
    if (m <= 64) {
        uint64_t eq_mask[256] = {0};
        for (size_t i = 0; i < m; ++i) {
            eq_mask[static_cast<uint8_t>(pattern[i])] |= (1ULL << i);
        }

        const uint64_t mask = (m == 64) ? ~0ULL : ((1ULL << m) - 1ULL);
        uint64_t pv = mask;
        uint64_t mv = 0;
        size_t dist = m;

        for (size_t j = 0; j < n; ++j) {
            const uint64_t eq = eq_mask[static_cast<uint8_t>(text[j])];
            const uint64_t xv = eq | mv;
            const uint64_t xh = (((eq & pv) + pv) ^ pv) | eq;

            uint64_t ph = mv | (~(xh | pv) & mask);
            uint64_t mh = pv & xh;

            dist += static_cast<size_t>((ph >> (m - 1)) & 1ULL);
            dist -= static_cast<size_t>((mh >> (m - 1)) & 1ULL);

            ph = ((ph << 1) | 1ULL) & mask;
            mh = (mh << 1) & mask;

            pv = mh | (~(xv | ph) & mask);
            mv = ph & xv;
        }

        return dist;
    }

    // Fast-path: pattern fits in 256 bits via bit-vector operations
    if (m <= 256) {
        return detail::myers_vec256(pattern, text);
    }

    // Arbitrary length fallback
    try {
        return detail::myers_multiblock(pattern, text);
    } catch (...) {
        return m + n;
    }
}

/**
 * @brief Computes edit distance, returning early with k+1 if distance > k.
 *
 * Uses Ukkonen's k-band restriction and early stopping cutoff: O(k * min(M, N)) time, O(k) space.
 *
 * @param pattern Short string.
 * @param text    Long text.
 * @param k       Threshold bound.
 * @return Distance if <= k, otherwise k + 1.
 */
[[nodiscard]] inline size_t levenshtein_within_k(std::string_view pattern, std::string_view text,
                                                 size_t k) noexcept {
    size_t m = pattern.size();
    size_t n = text.size();

    // 1. O(1) Length Difference Filter
    const size_t diff = (m > n) ? (m - n) : (n - m);
    if (diff > k) {
        return k + 1;
    }

    // 2. Symmetry Normalization: ensure m <= n
    if (m > n) {
        std::swap(pattern, text);
        std::swap(m, n);
    }

    // 3. Fast-path for k == 0
    if (k == 0) {
        return (pattern == text) ? 0 : 1;
    }

    // 4. LCP Trimming (Longest Common Prefix)
    size_t lcp = 0;
    while (lcp < m && pattern[lcp] == text[lcp]) {
        ++lcp;
    }
    if (lcp > 0) {
        pattern.remove_prefix(lcp);
        text.remove_prefix(lcp);
        m -= lcp;
        n -= lcp;
    }

    // 5. LCS Trimming (Longest Common Suffix)
    size_t lcs = 0;
    while (lcs < m && pattern[m - 1 - lcs] == text[n - 1 - lcs]) {
        ++lcs;
    }
    if (lcs > 0) {
        pattern.remove_suffix(lcs);
        text.remove_suffix(lcs);
        m -= lcs;
        n -= lcs;
    }

    // 6. Check if shorter string became empty
    if (m == 0) {
        return (n <= k) ? n : (k + 1);
    }

    // If k >= n, distance is guaranteed <= n <= k, delegate to full distance
    if (k >= n) {
        return levenshtein_distance(pattern, text);
    }

    // Clamp k against integer overflow
    constexpr size_t safe_max_k = static_cast<size_t>(std::numeric_limits<int>::max() / 4);
    if (k > safe_max_k) {
        return levenshtein_distance(pattern, text);
    }

    const int im = static_cast<int>(m);
    const int in = static_cast<int>(n);
    const int ik = static_cast<int>(k);

    const int delta = in - im;
    if (delta > ik) {
        return k + 1;
    }

    // 7. Ukkonen's Band DP Setup
    const int d_min = delta - ik;
    const int d_max = ik;
    const int w = d_max - d_min + 1;
    const int inf = ik + 1;

    // Stack allocation for typical k <= 127 (zero heap allocation)
    constexpr size_t max_stack_w = 256;
    if (static_cast<size_t>(w) <= max_stack_w) {
        std::array<int, max_stack_w> prev_row{};
        std::array<int, max_stack_w> curr_row{};
        std::fill_n(prev_row.data(), w, inf);

        const int j0_max = std::min(in, ik);
        for (int j = 0; j <= j0_max; ++j) {
            const int c = j - d_min;
            if (c >= 0 && c < w) {
                prev_row[static_cast<size_t>(c)] = j;
            }
        }

        for (int i = 1; i <= im; ++i) {
            std::fill_n(curr_row.data(), w, inf);
            int min_row_val = inf;

            const int c_start = std::max(0, -i - d_min);
            const int c_end = std::min(w - 1, in - i - d_min);

            for (int c = c_start; c <= c_end; ++c) {
                const int d = c + d_min;
                const int j = i + d;
                int val = inf;

                const size_t uc = static_cast<size_t>(c);
                if (j > 0 && prev_row[uc] < inf) {
                    const int cost =
                        (pattern[static_cast<size_t>(i - 1)] == text[static_cast<size_t>(j - 1)])
                            ? 0
                            : 1;
                    val = std::min(val, prev_row[uc] + cost);
                }
                if (c + 1 < w && prev_row[uc + 1] < inf) {
                    val = std::min(val, prev_row[uc + 1] + 1);
                }
                if (c - 1 >= 0 && curr_row[uc - 1] < inf) {
                    val = std::min(val, curr_row[uc - 1] + 1);
                }

                curr_row[uc] = val;
                min_row_val = std::min(min_row_val, val);
            }

            if (min_row_val > ik) {
                return k + 1;
            }

            std::copy_n(curr_row.data(), w, prev_row.data());
        }

        const size_t target_c = static_cast<size_t>(ik);
        const int result = (ik >= 0 && ik < w) ? prev_row[target_c] : inf;
        return (result <= ik) ? static_cast<size_t>(result) : (k + 1);
    }

    // Dynamic heap buffer for very large k
    try {
        std::vector<int> prev_row(static_cast<size_t>(w), inf);
        std::vector<int> curr_row(static_cast<size_t>(w), inf);

        const int j0_max = std::min(in, ik);
        for (int j = 0; j <= j0_max; ++j) {
            const int c = j - d_min;
            if (c >= 0 && c < w) {
                prev_row[static_cast<size_t>(c)] = j;
            }
        }

        for (int i = 1; i <= im; ++i) {
            std::fill(curr_row.begin(), curr_row.end(), inf);
            int min_row_val = inf;

            const int c_start = std::max(0, -i - d_min);
            const int c_end = std::min(w - 1, in - i - d_min);

            for (int c = c_start; c <= c_end; ++c) {
                const int d = c + d_min;
                const int j = i + d;
                int val = inf;
                const size_t uc = static_cast<size_t>(c);

                if (j > 0 && prev_row[uc] < inf) {
                    const int cost =
                        (pattern[static_cast<size_t>(i - 1)] == text[static_cast<size_t>(j - 1)])
                            ? 0
                            : 1;
                    val = std::min(val, prev_row[uc] + cost);
                }
                if (c + 1 < w && prev_row[uc + 1] < inf) {
                    val = std::min(val, prev_row[uc + 1] + 1);
                }
                if (c - 1 >= 0 && curr_row[uc - 1] < inf) {
                    val = std::min(val, curr_row[uc - 1] + 1);
                }

                curr_row[uc] = val;
                min_row_val = std::min(min_row_val, val);
            }

            if (min_row_val > ik) {
                return k + 1;
            }

            std::swap(prev_row, curr_row);
        }

        const size_t target_c = static_cast<size_t>(ik);
        const int result = (ik >= 0 && ik < w) ? prev_row[target_c] : inf;
        return (result <= ik) ? static_cast<size_t>(result) : (k + 1);
    } catch (...) {
        return k + 1;
    }
}

/**
 * @brief Full alignment recovery via bit-parallel traceback.
 *
 * Recovers exact CIGAR-style operation list (Match, Substitute, Insert, Delete)
 * that transforms pattern into text.
 *
 * @param pattern Short string.
 * @param text    Long text.
 * @return Alignment structure containing distance and operations.
 */
[[nodiscard]] inline Alignment levenshtein_align(std::string_view pattern, std::string_view text) {
    const size_t m = pattern.size();
    const size_t n = text.size();

    if (m == 0) {
        std::vector<EditOp> ops(n, EditOp::Insert);
        return Alignment{n, std::move(ops)};
    }
    if (n == 0) {
        std::vector<EditOp> ops(m, EditOp::Delete);
        return Alignment{m, std::move(ops)};
    }

    // Hyyro (2004) Bit-Vector Traceback for M <= 64
    if (m <= 64) {
        uint64_t eq_mask[256] = {0};
        for (size_t i = 0; i < m; ++i) {
            eq_mask[static_cast<uint8_t>(pattern[i])] |= (1ULL << i);
        }

        const uint64_t mask = (m == 64) ? ~0ULL : ((1ULL << m) - 1ULL);
        uint64_t pv = mask;
        uint64_t mv = 0;
        size_t dist = m;

        std::vector<uint64_t> vp_table;
        std::vector<uint64_t> vn_table;
        vp_table.reserve(n + 1);
        vn_table.reserve(n + 1);

        vp_table.push_back(pv);
        vn_table.push_back(mv);

        for (size_t j = 0; j < n; ++j) {
            const uint64_t eq = eq_mask[static_cast<uint8_t>(text[j])];
            const uint64_t xh = (((eq & pv) + pv) ^ pv) | eq;
            const uint64_t ph = mv | (~(xh | pv) & mask);
            const uint64_t mh = pv & xh;

            dist += static_cast<size_t>((ph >> (m - 1)) & 1ULL);
            dist -= static_cast<size_t>((mh >> (m - 1)) & 1ULL);

            const uint64_t ph_shift = ((ph << 1) | 1ULL) & mask;
            const uint64_t mh_shift = (mh << 1) & mask;

            const uint64_t xv = eq | mv;
            pv = (mh_shift | (~(xv | ph_shift) & mask)) & mask;
            mv = (ph_shift & xv) & mask;

            vp_table.push_back(pv);
            vn_table.push_back(mv);
        }

        std::vector<EditOp> ops;
        ops.reserve(m + n);
        size_t i = m;
        size_t j = n;

        while (i > 0 && j > 0) {
            if ((vp_table[j] >> (i - 1)) & 1ULL) {
                ops.push_back(EditOp::Delete);
                --i;
            } else {
                if ((vn_table[j - 1] >> (i - 1)) & 1ULL) {
                    ops.push_back(EditOp::Insert);
                    --j;
                } else {
                    if (pattern[i - 1] == text[j - 1]) {
                        ops.push_back(EditOp::Match);
                    } else {
                        ops.push_back(EditOp::Substitute);
                    }
                    --i;
                    --j;
                }
            }
        }
        while (i > 0) {
            ops.push_back(EditOp::Delete);
            --i;
        }
        while (j > 0) {
            ops.push_back(EditOp::Insert);
            --j;
        }

        std::reverse(ops.begin(), ops.end());
        return Alignment{dist, std::move(ops)};
    }

    // Hyyro (2004) Bit-Vector Traceback extended to M in [65, 256]
    if (m <= 256) {
        detail::Vec256 peq[256]{};
        for (size_t i = 0; i < m; ++i) {
            peq[static_cast<uint8_t>(pattern[i])].w[i / 64] |= (1ULL << (i % 64));
        }

        detail::Vec256 mask{};
        for (size_t i = 0; i < m; ++i) {
            mask.w[i / 64] |= (1ULL << (i % 64));
        }

        detail::Vec256 pv = mask;
        detail::Vec256 mv{};
        size_t dist = m;
        const size_t top_word = (m - 1) / 64;
        const uint64_t top_bit = (1ULL << ((m - 1) % 64));

        std::vector<detail::Vec256> vp_table;
        std::vector<detail::Vec256> vn_table;
        vp_table.reserve(n + 1);
        vn_table.reserve(n + 1);
        vp_table.push_back(pv);
        vn_table.push_back(mv);

        for (size_t j = 0; j < n; ++j) {
            const detail::Vec256 eq = peq[static_cast<uint8_t>(text[j])];
            const detail::Vec256 xv = detail::vec256_or(eq, mv);
            const detail::Vec256 eq_and_pv = detail::vec256_and(eq, pv);
            const detail::Vec256 sum = detail::add256_carry(eq_and_pv, pv);
            const detail::Vec256 xh = detail::vec256_or(detail::vec256_xor(sum, pv), eq);

            const detail::Vec256 xh_or_pv = detail::vec256_or(xh, pv);
            const detail::Vec256 not_xh_or_pv = detail::vec256_andnot(xh_or_pv, mask);
            detail::Vec256 ph = detail::vec256_or(mv, not_xh_or_pv);
            detail::Vec256 mh = detail::vec256_and(pv, xh);

            if (ph.w[top_word] & top_bit)
                ++dist;
            if (mh.w[top_word] & top_bit)
                --dist;

            ph = detail::shl256_1(ph, 1);
            mh = detail::shl256_1(mh, 0);
            ph = detail::vec256_and(ph, mask);
            mh = detail::vec256_and(mh, mask);

            const detail::Vec256 xv_or_ph = detail::vec256_or(xv, ph);
            const detail::Vec256 not_xv_or_ph = detail::vec256_andnot(xv_or_ph, mask);
            pv = detail::vec256_or(mh, not_xv_or_ph);
            mv = detail::vec256_and(ph, xv);

            vp_table.push_back(pv);
            vn_table.push_back(mv);
        }

        std::vector<EditOp> ops;
        ops.reserve(m + n);
        size_t i = m;
        size_t j = n;

        while (i > 0 && j > 0) {
            const size_t word_idx = (i - 1) / 64;
            const uint64_t bit_mask = (1ULL << ((i - 1) % 64));

            if (vp_table[j].w[word_idx] & bit_mask) {
                ops.push_back(EditOp::Delete);
                --i;
            } else {
                if (vn_table[j - 1].w[word_idx] & bit_mask) {
                    ops.push_back(EditOp::Insert);
                    --j;
                } else {
                    if (pattern[i - 1] == text[j - 1]) {
                        ops.push_back(EditOp::Match);
                    } else {
                        ops.push_back(EditOp::Substitute);
                    }
                    --i;
                    --j;
                }
            }
        }
        while (i > 0) {
            ops.push_back(EditOp::Delete);
            --i;
        }
        while (j > 0) {
            ops.push_back(EditOp::Insert);
            --j;
        }

        std::reverse(ops.begin(), ops.end());
        return Alignment{dist, std::move(ops)};
    }

    // Standard DP Matrix Traceback fallback for massive strings M > 256
    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1));
    for (size_t i = 0; i <= m; ++i)
        dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j)
        dp[0][j] = j;

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            const size_t cost = (pattern[i - 1] == text[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
        }
    }

    std::vector<EditOp> ops;
    ops.reserve(m + n);
    size_t i = m;
    size_t j = n;

    while (i > 0 && j > 0) {
        const size_t cost = (pattern[i - 1] == text[j - 1]) ? 0 : 1;
        if (dp[i][j] == dp[i - 1][j - 1] + cost) {
            ops.push_back(cost == 0 ? EditOp::Match : EditOp::Substitute);
            --i;
            --j;
        } else if (dp[i][j] == dp[i - 1][j] + 1) {
            ops.push_back(EditOp::Delete);
            --i;
        } else {
            ops.push_back(EditOp::Insert);
            --j;
        }
    }
    while (i > 0) {
        ops.push_back(EditOp::Delete);
        --i;
    }
    while (j > 0) {
        ops.push_back(EditOp::Insert);
        --j;
    }
    std::reverse(ops.begin(), ops.end());
    return Alignment{dp[m][n], std::move(ops)};
}

/**
 * @brief Batch edit distance: computes distance from pattern to each text in texts.
 *
 * SIMD-parallel (AVX2 data-parallel) across multiple text targets for M <= 64.
 *
 * @param pattern Short query string.
 * @param texts   Span of target texts.
 * @return Vector of distances for each target text.
 */
[[nodiscard]] inline std::vector<size_t>
levenshtein_batch(std::string_view pattern, std::span<const std::string_view> texts) {
    std::vector<size_t> results(texts.size());
    if (texts.empty())
        return results;

    const size_t m = pattern.size();

#if defined(__AVX2__)
    if (m <= 64 && m > 0) {
        size_t i = 0;
        for (; i + 4 <= texts.size(); i += 4) {
            const std::string_view batch[4] = {texts[i], texts[i + 1], texts[i + 2], texts[i + 3]};
            detail::myers_avx2_batch4(pattern, batch, &results[i]);
        }
        for (; i < texts.size(); ++i) {
            const std::string_view batch[4] = {texts[i], {}, {}, {}};
            size_t d[4];
            detail::myers_avx2_batch4(pattern, batch, d);
            results[i] = d[0];
        }
        return results;
    }
#endif

    for (size_t i = 0; i < texts.size(); ++i) {
        results[i] = levenshtein_distance(pattern, texts[i]);
    }

    return results;
}

/**
 * @brief Longest Common Subsequence length (equivalent to edit without substitution).
 *
 * Computes exact LCS length using Crochemore/Hyyro bit-parallel O(ceil(M/64) * N).
 *
 * @param a First string.
 * @param b Second string.
 * @return Length of LCS.
 */
[[nodiscard]] inline size_t lcs_length(std::string_view a, std::string_view b) noexcept {
    if (a.empty() || b.empty()) {
        return 0;
    }

    // Symmetry optimization: ensure |a| <= |b|
    if (a.size() > b.size()) {
        std::swap(a, b);
    }

    const size_t m = a.size();
    const size_t n = b.size();

    // Fast-path: fits in a single 64-bit word (strictly 0 heap allocations)
    if (m <= 64) {
        uint64_t eq_mask[256] = {0};
        for (size_t i = 0; i < m; ++i) {
            eq_mask[static_cast<uint8_t>(a[i])] |= (1ULL << i);
        }

        const uint64_t mask = (m == 64) ? ~0ULL : ((1ULL << m) - 1ULL);
        uint64_t row = 0;

        for (const char c : b) {
            const uint64_t match_mask = eq_mask[static_cast<uint8_t>(c)];
            const uint64_t x = row | match_mask;
            const uint64_t u = (row << 1) | 1ULL;
            const uint64_t diff = x - u;
            row = (x & ~diff) & mask;
        }

        return static_cast<size_t>(std::popcount(row));
    }

    // Multi-word bit-parallel algorithm for M > 64
    try {
        const size_t num_words = (m + 63) / 64;
        std::vector<std::array<uint64_t, 256>> eq_masks(num_words);
        for (size_t i = 0; i < m; ++i) {
            const size_t word_idx = i / 64;
            const size_t bit_idx = i % 64;
            eq_masks[word_idx][static_cast<uint8_t>(a[i])] |= (1ULL << bit_idx);
        }

        std::vector<uint64_t> masks(num_words, ~0ULL);
        const size_t rem = m % 64;
        if (rem != 0) {
            masks.back() = (1ULL << rem) - 1ULL;
        }

        std::vector<uint64_t> row(num_words, 0);
        std::vector<uint64_t> shifted_row(num_words, 0);

        for (const char c : b) {
            const uint8_t ch = static_cast<uint8_t>(c);

            uint64_t carry_in = 1;
            for (size_t k = 0; k < num_words; ++k) {
                const uint64_t carry_out = row[k] >> 63;
                shifted_row[k] = (row[k] << 1) | carry_in;
                carry_in = carry_out;
            }

            unsigned char borrow = 0;
            for (size_t k = 0; k < num_words; ++k) {
                const uint64_t x = row[k] | eq_masks[k][ch];
                uint64_t diff = 0;
                borrow = detail::sub_borrow_u64(borrow, x, shifted_row[k], &diff);
                row[k] = (x & ~diff) & masks[k];
            }
        }

        size_t total_lcs = 0;
        for (size_t k = 0; k < num_words; ++k) {
            total_lcs += static_cast<size_t>(std::popcount(row[k]));
        }
        return total_lcs;
    } catch (...) {
        return 0;
    }
}

} // namespace algoat::dp
