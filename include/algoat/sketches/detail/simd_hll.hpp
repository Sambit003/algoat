#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace algoat::sketches::detail {

/// Compile-time lookup table for scalar 2^-k powers in double precision
[[nodiscard]] constexpr auto make_power_of_two_lut() noexcept {
    struct LUT {
        double val[64];
    };
    LUT lut{};
    for (int i = 0; i < 64; ++i) {
        // IEEE-754 binary64 bit pattern for 2^-i
        uint64_t bits = static_cast<uint64_t>(1023 - i) << 52;
        // In constexpr C++20, std::bit_cast gives bit exact float64
        lut.val[i] = std::bit_cast<double>(bits);
    }
    return lut;
}

inline constexpr auto POWER_OF_TWO_LUT = make_power_of_two_lut();

/**
 * @brief Evaluate harmonic sum S = sum_{j=0}^{m-1} 2^{-M[j]} with AVX2 vectorization.
 *
 * NOTE ON PRECISION:
 * Double precision (binary64, 53-bit mantissa) is intentionally chosen over single precision
 * (binary32, 24-bit mantissa). At high precisions like P = 18 (m = 262,144), summing 262,144 terms
 * in float32 causes catastrophic floating-point absorption: the partial sum (~4,000) exceeds small
 * register contributions by 2^18, resulting in 1.56% numerical error (7.7x larger than the
 * theoretical standard error of 0.203%). Binary64 accumulation guarantees zero numerical drift.
 */
[[nodiscard]] inline double simd_harmonic_sum(const uint8_t* registers, std::size_t m) noexcept {
    double total = 0.0;
    std::size_t i = 0;

#if defined(__AVX2__)
    // 4 parallel vector accumulators (16 doubles per step)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    const __m256i bias = _mm256_set1_epi64x(1023);
    const std::size_t vector_end = m & ~static_cast<std::size_t>(15);

    for (; i < vector_end; i += 16) {
        // Load 16 bytes into lower 128-bit lane
        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(registers + i));

        // Group 0: bytes 0..3 -> 4x int64
        __m256i k0 = _mm256_cvtepu8_epi64(raw16);
        __m256i exp0 = _mm256_sub_epi64(bias, k0);
        __m256d f0 = _mm256_castsi256_pd(_mm256_slli_epi64(exp0, 52));
        acc0 = _mm256_add_pd(acc0, f0);

        // Group 1: bytes 4..7 -> 4x int64
        __m128i raw_hi4 = _mm_srli_si128(raw16, 4);
        __m256i k1 = _mm256_cvtepu8_epi64(raw_hi4);
        __m256i exp1 = _mm256_sub_epi64(bias, k1);
        __m256d f1 = _mm256_castsi256_pd(_mm256_slli_epi64(exp1, 52));
        acc1 = _mm256_add_pd(acc1, f1);

        // Group 2: bytes 8..11 -> 4x int64
        __m128i raw_hi8 = _mm_srli_si128(raw16, 8);
        __m256i k2 = _mm256_cvtepu8_epi64(raw_hi8);
        __m256i exp2 = _mm256_sub_epi64(bias, k2);
        __m256d f2 = _mm256_castsi256_pd(_mm256_slli_epi64(exp2, 52));
        acc2 = _mm256_add_pd(acc2, f2);

        // Group 3: bytes 12..15 -> 4x int64
        __m128i raw_hi12 = _mm_srli_si128(raw16, 12);
        __m256i k3 = _mm256_cvtepu8_epi64(raw_hi12);
        __m256i exp3 = _mm256_sub_epi64(bias, k3);
        __m256d f3 = _mm256_castsi256_pd(_mm256_slli_epi64(exp3, 52));
        acc3 = _mm256_add_pd(acc3, f3);
    }

    // Combine 4 vector accumulators
    __m256d sum256 = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    alignas(32) double buf[4];
    _mm256_store_pd(buf, sum256);
    total = (buf[0] + buf[1]) + (buf[2] + buf[3]);
#endif

    // Unified scalar tail / fallback
    for (; i < m; ++i) {
        total += POWER_OF_TWO_LUT.val[registers[i] & 0x3F];
    }
    return total;
}

/// Compute M_dst[j] = max(M_dst[j], M_src[j]) with AVX2 SIMD acceleration
inline void simd_dense_merge(uint8_t* dst, const uint8_t* src, std::size_t m) noexcept {
    std::size_t i = 0;

#if defined(__AVX2__)
    const std::size_t vector_end = m & ~static_cast<std::size_t>(127);

    for (; i < vector_end; i += 128) {
        __m256i d0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        __m256i s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_max_epu8(d0, s0));

        __m256i d1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i + 32));
        __m256i s1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 32));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 32), _mm256_max_epu8(d1, s1));

        __m256i d2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i + 64));
        __m256i s2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 64));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 64), _mm256_max_epu8(d2, s2));

        __m256i d3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i + 96));
        __m256i s3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 96));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 96), _mm256_max_epu8(d3, s3));
    }
#endif

    // Unified scalar tail / fallback
    for (; i < m; ++i) {
        dst[i] = std::max(dst[i], src[i]);
    }
}

/// Count number of registers equal to 0 with AVX2 SIMD acceleration
[[nodiscard]] inline std::size_t simd_count_zeros(const uint8_t* registers,
                                                  std::size_t m) noexcept {
    std::size_t count = 0;
    std::size_t i = 0;

#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const std::size_t vector_end = m & ~static_cast<std::size_t>(31);

    for (; i < vector_end; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(registers + i));
        __m256i cmp = _mm256_cmpeq_epi8(v, zero);
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));
        count += static_cast<std::size_t>(std::popcount(mask));
    }
#endif

    // Unified scalar tail / fallback
    for (; i < m; ++i) {
        if (registers[i] == 0) {
            count++;
        }
    }
    return count;
}

} // namespace algoat::sketches::detail
