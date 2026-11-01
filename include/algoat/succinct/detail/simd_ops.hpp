#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace algoat::succinct::detail {

inline constexpr size_t BITSET_WORDS = 1024;
inline constexpr size_t BITSET_BYTES = BITSET_WORDS * sizeof(uint64_t); // 8192 bytes

struct SimdOps {
    static void bitset_and(uint64_t* out, const uint64_t* a, const uint64_t* b) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        const auto* va = reinterpret_cast<const __m512i*>(a);
        const auto* vb = reinterpret_cast<const __m512i*>(b);
        auto* vout = reinterpret_cast<__m512i*>(out);
        for (size_t i = 0; i < 128; i += 4) {
            _mm512_storeu_si512(vout + i + 0, _mm512_and_si512(_mm512_loadu_si512(va + i + 0),
                                                               _mm512_loadu_si512(vb + i + 0)));
            _mm512_storeu_si512(vout + i + 1, _mm512_and_si512(_mm512_loadu_si512(va + i + 1),
                                                               _mm512_loadu_si512(vb + i + 1)));
            _mm512_storeu_si512(vout + i + 2, _mm512_and_si512(_mm512_loadu_si512(va + i + 2),
                                                               _mm512_loadu_si512(vb + i + 2)));
            _mm512_storeu_si512(vout + i + 3, _mm512_and_si512(_mm512_loadu_si512(va + i + 3),
                                                               _mm512_loadu_si512(vb + i + 3)));
        }
#elif defined(__AVX2__)
        const auto* va = reinterpret_cast<const __m256i*>(a);
        const auto* vb = reinterpret_cast<const __m256i*>(b);
        auto* vout = reinterpret_cast<__m256i*>(out);
        for (size_t i = 0; i < 256; i += 8) {
            _mm256_storeu_si256(vout + i + 0, _mm256_and_si256(_mm256_loadu_si256(va + i + 0),
                                                               _mm256_loadu_si256(vb + i + 0)));
            _mm256_storeu_si256(vout + i + 1, _mm256_and_si256(_mm256_loadu_si256(va + i + 1),
                                                               _mm256_loadu_si256(vb + i + 1)));
            _mm256_storeu_si256(vout + i + 2, _mm256_and_si256(_mm256_loadu_si256(va + i + 2),
                                                               _mm256_loadu_si256(vb + i + 2)));
            _mm256_storeu_si256(vout + i + 3, _mm256_and_si256(_mm256_loadu_si256(va + i + 3),
                                                               _mm256_loadu_si256(vb + i + 3)));
            _mm256_storeu_si256(vout + i + 4, _mm256_and_si256(_mm256_loadu_si256(va + i + 4),
                                                               _mm256_loadu_si256(vb + i + 4)));
            _mm256_storeu_si256(vout + i + 5, _mm256_and_si256(_mm256_loadu_si256(va + i + 5),
                                                               _mm256_loadu_si256(vb + i + 5)));
            _mm256_storeu_si256(vout + i + 6, _mm256_and_si256(_mm256_loadu_si256(va + i + 6),
                                                               _mm256_loadu_si256(vb + i + 6)));
            _mm256_storeu_si256(vout + i + 7, _mm256_and_si256(_mm256_loadu_si256(va + i + 7),
                                                               _mm256_loadu_si256(vb + i + 7)));
        }
#else
        for (size_t i = 0; i < BITSET_WORDS; i += 4) {
            out[i + 0] = a[i + 0] & b[i + 0];
            out[i + 1] = a[i + 1] & b[i + 1];
            out[i + 2] = a[i + 2] & b[i + 2];
            out[i + 3] = a[i + 3] & b[i + 3];
        }
#endif
    }

    static void bitset_or(uint64_t* out, const uint64_t* a, const uint64_t* b) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        const auto* va = reinterpret_cast<const __m512i*>(a);
        const auto* vb = reinterpret_cast<const __m512i*>(b);
        auto* vout = reinterpret_cast<__m512i*>(out);
        for (size_t i = 0; i < 128; i += 4) {
            _mm512_storeu_si512(vout + i + 0, _mm512_or_si512(_mm512_loadu_si512(va + i + 0),
                                                              _mm512_loadu_si512(vb + i + 0)));
            _mm512_storeu_si512(vout + i + 1, _mm512_or_si512(_mm512_loadu_si512(va + i + 1),
                                                              _mm512_loadu_si512(vb + i + 1)));
            _mm512_storeu_si512(vout + i + 2, _mm512_or_si512(_mm512_loadu_si512(va + i + 2),
                                                              _mm512_loadu_si512(vb + i + 2)));
            _mm512_storeu_si512(vout + i + 3, _mm512_or_si512(_mm512_loadu_si512(va + i + 3),
                                                              _mm512_loadu_si512(vb + i + 3)));
        }
#elif defined(__AVX2__)
        const auto* va = reinterpret_cast<const __m256i*>(a);
        const auto* vb = reinterpret_cast<const __m256i*>(b);
        auto* vout = reinterpret_cast<__m256i*>(out);
        for (size_t i = 0; i < 256; i += 8) {
            _mm256_storeu_si256(vout + i + 0, _mm256_or_si256(_mm256_loadu_si256(va + i + 0),
                                                              _mm256_loadu_si256(vb + i + 0)));
            _mm256_storeu_si256(vout + i + 1, _mm256_or_si256(_mm256_loadu_si256(va + i + 1),
                                                              _mm256_loadu_si256(vb + i + 1)));
            _mm256_storeu_si256(vout + i + 2, _mm256_or_si256(_mm256_loadu_si256(va + i + 2),
                                                              _mm256_loadu_si256(vb + i + 2)));
            _mm256_storeu_si256(vout + i + 3, _mm256_or_si256(_mm256_loadu_si256(va + i + 3),
                                                              _mm256_loadu_si256(vb + i + 3)));
            _mm256_storeu_si256(vout + i + 4, _mm256_or_si256(_mm256_loadu_si256(va + i + 4),
                                                              _mm256_loadu_si256(vb + i + 4)));
            _mm256_storeu_si256(vout + i + 5, _mm256_or_si256(_mm256_loadu_si256(va + i + 5),
                                                              _mm256_loadu_si256(vb + i + 5)));
            _mm256_storeu_si256(vout + i + 6, _mm256_or_si256(_mm256_loadu_si256(va + i + 6),
                                                              _mm256_loadu_si256(vb + i + 6)));
            _mm256_storeu_si256(vout + i + 7, _mm256_or_si256(_mm256_loadu_si256(va + i + 7),
                                                              _mm256_loadu_si256(vb + i + 7)));
        }
#else
        for (size_t i = 0; i < BITSET_WORDS; i += 4) {
            out[i + 0] = a[i + 0] | b[i + 0];
            out[i + 1] = a[i + 1] | b[i + 1];
            out[i + 2] = a[i + 2] | b[i + 2];
            out[i + 3] = a[i + 3] | b[i + 3];
        }
#endif
    }

    static void bitset_xor(uint64_t* out, const uint64_t* a, const uint64_t* b) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        const auto* va = reinterpret_cast<const __m512i*>(a);
        const auto* vb = reinterpret_cast<const __m512i*>(b);
        auto* vout = reinterpret_cast<__m512i*>(out);
        for (size_t i = 0; i < 128; i += 4) {
            _mm512_storeu_si512(vout + i + 0, _mm512_xor_si512(_mm512_loadu_si512(va + i + 0),
                                                               _mm512_loadu_si512(vb + i + 0)));
            _mm512_storeu_si512(vout + i + 1, _mm512_xor_si512(_mm512_loadu_si512(va + i + 1),
                                                               _mm512_loadu_si512(vb + i + 1)));
            _mm512_storeu_si512(vout + i + 2, _mm512_xor_si512(_mm512_loadu_si512(va + i + 2),
                                                               _mm512_loadu_si512(vb + i + 2)));
            _mm512_storeu_si512(vout + i + 3, _mm512_xor_si512(_mm512_loadu_si512(va + i + 3),
                                                               _mm512_loadu_si512(vb + i + 3)));
        }
#elif defined(__AVX2__)
        const auto* va = reinterpret_cast<const __m256i*>(a);
        const auto* vb = reinterpret_cast<const __m256i*>(b);
        auto* vout = reinterpret_cast<__m256i*>(out);
        for (size_t i = 0; i < 256; i += 8) {
            _mm256_storeu_si256(vout + i + 0, _mm256_xor_si256(_mm256_loadu_si256(va + i + 0),
                                                               _mm256_loadu_si256(vb + i + 0)));
            _mm256_storeu_si256(vout + i + 1, _mm256_xor_si256(_mm256_loadu_si256(va + i + 1),
                                                               _mm256_loadu_si256(vb + i + 1)));
            _mm256_storeu_si256(vout + i + 2, _mm256_xor_si256(_mm256_loadu_si256(va + i + 2),
                                                               _mm256_loadu_si256(vb + i + 2)));
            _mm256_storeu_si256(vout + i + 3, _mm256_xor_si256(_mm256_loadu_si256(va + i + 3),
                                                               _mm256_loadu_si256(vb + i + 3)));
            _mm256_storeu_si256(vout + i + 4, _mm256_xor_si256(_mm256_loadu_si256(va + i + 4),
                                                               _mm256_loadu_si256(vb + i + 4)));
            _mm256_storeu_si256(vout + i + 5, _mm256_xor_si256(_mm256_loadu_si256(va + i + 5),
                                                               _mm256_loadu_si256(vb + i + 5)));
            _mm256_storeu_si256(vout + i + 6, _mm256_xor_si256(_mm256_loadu_si256(va + i + 6),
                                                               _mm256_loadu_si256(vb + i + 6)));
            _mm256_storeu_si256(vout + i + 7, _mm256_xor_si256(_mm256_loadu_si256(va + i + 7),
                                                               _mm256_loadu_si256(vb + i + 7)));
        }
#else
        for (size_t i = 0; i < BITSET_WORDS; i += 4) {
            out[i + 0] = a[i + 0] ^ b[i + 0];
            out[i + 1] = a[i + 1] ^ b[i + 1];
            out[i + 2] = a[i + 2] ^ b[i + 2];
            out[i + 3] = a[i + 3] ^ b[i + 3];
        }
#endif
    }

    // Computes a & ~b
    static void bitset_andnot(uint64_t* out, const uint64_t* a, const uint64_t* b) noexcept {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        const auto* va = reinterpret_cast<const __m512i*>(a);
        const auto* vb = reinterpret_cast<const __m512i*>(b);
        auto* vout = reinterpret_cast<__m512i*>(out);
        for (size_t i = 0; i < 128; i += 4) {
            _mm512_storeu_si512(vout + i + 0, _mm512_andnot_si512(_mm512_loadu_si512(vb + i + 0),
                                                                  _mm512_loadu_si512(va + i + 0)));
            _mm512_storeu_si512(vout + i + 1, _mm512_andnot_si512(_mm512_loadu_si512(vb + i + 1),
                                                                  _mm512_loadu_si512(va + i + 1)));
            _mm512_storeu_si512(vout + i + 2, _mm512_andnot_si512(_mm512_loadu_si512(vb + i + 2),
                                                                  _mm512_loadu_si512(va + i + 2)));
            _mm512_storeu_si512(vout + i + 3, _mm512_andnot_si512(_mm512_loadu_si512(vb + i + 3),
                                                                  _mm512_loadu_si512(va + i + 3)));
        }
#elif defined(__AVX2__)
        const auto* va = reinterpret_cast<const __m256i*>(a);
        const auto* vb = reinterpret_cast<const __m256i*>(b);
        auto* vout = reinterpret_cast<__m256i*>(out);
        for (size_t i = 0; i < 256; i += 8) {
            _mm256_storeu_si256(vout + i + 0, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 0),
                                                                  _mm256_loadu_si256(va + i + 0)));
            _mm256_storeu_si256(vout + i + 1, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 1),
                                                                  _mm256_loadu_si256(va + i + 1)));
            _mm256_storeu_si256(vout + i + 2, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 2),
                                                                  _mm256_loadu_si256(va + i + 2)));
            _mm256_storeu_si256(vout + i + 3, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 3),
                                                                  _mm256_loadu_si256(va + i + 3)));
            _mm256_storeu_si256(vout + i + 4, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 4),
                                                                  _mm256_loadu_si256(va + i + 4)));
            _mm256_storeu_si256(vout + i + 5, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 5),
                                                                  _mm256_loadu_si256(va + i + 5)));
            _mm256_storeu_si256(vout + i + 6, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 6),
                                                                  _mm256_loadu_si256(va + i + 6)));
            _mm256_storeu_si256(vout + i + 7, _mm256_andnot_si256(_mm256_loadu_si256(vb + i + 7),
                                                                  _mm256_loadu_si256(va + i + 7)));
        }
#else
        for (size_t i = 0; i < BITSET_WORDS; i += 4) {
            out[i + 0] = a[i + 0] & ~b[i + 0];
            out[i + 1] = a[i + 1] & ~b[i + 1];
            out[i + 2] = a[i + 2] & ~b[i + 2];
            out[i + 3] = a[i + 3] & ~b[i + 3];
        }
#endif
    }

    static uint64_t bitset_popcount(const uint64_t* words) noexcept {
#if defined(__AVX512VPOPCNTDQ__)
        const auto* vwords = reinterpret_cast<const __m512i*>(words);
        __m512i vtotal = _mm512_setzero_si512();
        for (size_t i = 0; i < 128; ++i) {
            vtotal = _mm512_add_epi64(vtotal, _mm512_popcnt_epi64(_mm512_loadu_si512(vwords + i)));
        }
        alignas(64) uint64_t res[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(res), vtotal);
        return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
#else
        uint64_t sum0 = 0;
        uint64_t sum1 = 0;
        uint64_t sum2 = 0;
        uint64_t sum3 = 0;
        for (size_t i = 0; i < BITSET_WORDS; i += 4) {
            sum0 += static_cast<uint64_t>(std::popcount(words[i + 0]));
            sum1 += static_cast<uint64_t>(std::popcount(words[i + 1]));
            sum2 += static_cast<uint64_t>(std::popcount(words[i + 2]));
            sum3 += static_cast<uint64_t>(std::popcount(words[i + 3]));
        }
        return sum0 + sum1 + sum2 + sum3;
#endif
    }

    static uint64_t bitset_and_cardinality(const uint64_t* a, const uint64_t* b) noexcept {
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)
        const auto* va = reinterpret_cast<const __m512i*>(a);
        const auto* vb = reinterpret_cast<const __m512i*>(b);
        __m512i vtotal = _mm512_setzero_si512();
        for (size_t i = 0; i < 128; ++i) {
            __m512i vres = _mm512_and_si512(_mm512_loadu_si512(va + i), _mm512_loadu_si512(vb + i));
            vtotal = _mm512_add_epi64(vtotal, _mm512_popcnt_epi64(vres));
        }
        alignas(64) uint64_t res[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(res), vtotal);
        return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
#else
        uint64_t sum0 = 0;
        uint64_t sum1 = 0;
        uint64_t sum2 = 0;
        uint64_t sum3 = 0;
        for (size_t i = 0; i < BITSET_WORDS; i += 4) {
            sum0 += static_cast<uint64_t>(std::popcount(a[i + 0] & b[i + 0]));
            sum1 += static_cast<uint64_t>(std::popcount(a[i + 1] & b[i + 1]));
            sum2 += static_cast<uint64_t>(std::popcount(a[i + 2] & b[i + 2]));
            sum3 += static_cast<uint64_t>(std::popcount(a[i + 3] & b[i + 3]));
        }
        return sum0 + sum1 + sum2 + sum3;
#endif
    }
};

} // namespace algoat::succinct::detail
