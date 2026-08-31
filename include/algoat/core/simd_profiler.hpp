/**
 * @file simd_profiler.hpp
 * @brief Sub-linear cache-line-aware SIMD array profiler using Google Highway and native
 * micro-helpers.
 */

#pragma once

#include <hwy/highway.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace algoat::core::detail {

namespace hn = hwy::HWY_NAMESPACE;

// Determine cache line size based on target architecture
#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
inline constexpr std::size_t kCacheLineBytes = 128; // Apple Silicon M-series (128-byte cache lines)
#else
inline constexpr std::size_t kCacheLineBytes =
    64; // Standard x86-64 and standard ARM (64-byte cache lines)
#endif

inline constexpr std::size_t kSublinearThreshold = 10000;
inline constexpr std::size_t kNumSampleChunks = 64;

/**
 * @brief Evaluates adjacent sortedness and equality across a contiguous memory chunk using SIMD.
 */
template <typename T>
HWY_INLINE void profile_chunk_simd(const T* ptr, std::size_t len, std::size_t& out_sorted_pairs,
                                   bool& out_has_duplicates) {
    if (len <= 1)
        return;

    const hn::ScalableTag<T> d;
    const std::size_t N = hn::Lanes(d);

    std::size_t i = 0;
    // Process full SIMD vector widths
    if (N > 0) {
        for (; i + N < len; i += N) {
            auto v1 = hn::LoadU(d, ptr + i);
            auto v2 = hn::LoadU(d, ptr + i + 1);

#if HWY_TARGET <= HWY_AVX3 && (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)
            // Explicit AVX-512 micro-helper escape hatch for 32-bit integers
            if constexpr (std::is_same_v<T, int32_t>) {
                __m512i r1 = v1.raw;
                __m512i r2 = v2.raw;
                __mmask16 le_mask = _mm512_cmple_epi32_mask(r1, r2);
                __mmask16 eq_mask = _mm512_cmpeq_epi32_mask(r1, r2);
                out_sorted_pairs +=
                    static_cast<std::size_t>(_mm_popcnt_u32(static_cast<uint32_t>(le_mask)));
                if (eq_mask != 0) {
                    out_has_duplicates = true;
                }
                continue;
            }
#endif

            auto le_mask = hn::Le(v1, v2);
            auto eq_mask = hn::Eq(v1, v2);
            out_sorted_pairs += hn::CountTrue(d, le_mask);
            if (hn::CountTrue(d, eq_mask) > 0) {
                out_has_duplicates = true;
            }
        }
    }

    // Process remaining scalar tail in chunk
    for (; i + 1 < len; ++i) {
        if (ptr[i] <= ptr[i + 1]) {
            out_sorted_pairs++;
        }
        if (ptr[i] == ptr[i + 1]) {
            out_has_duplicates = true;
        }
    }
}

/**
 * @brief Stratified cache-line-aware sub-linear array profiler.
 */
template <typename T>
inline void sample_traits_sublinear_impl(const T* data, std::size_t size,
                                         double& out_sortedness_ratio, bool& out_has_duplicates) {
    std::size_t chunk_elements = std::max<std::size_t>(kCacheLineBytes / sizeof(T), 16);
    std::size_t num_chunks = std::min<std::size_t>(kNumSampleChunks, size / chunk_elements);
    if (num_chunks == 0)
        num_chunks = 1;

    std::size_t total_sampled_pairs = 0;
    std::size_t sorted_sampled_pairs = 0;
    bool has_duplicates = false;

    // Check endpoints
    if (data[0] <= data[size - 1]) {
        sorted_sampled_pairs++;
    }
    total_sampled_pairs++;

    // Calculate stride across the array
    std::size_t stride = (size > chunk_elements) ? (size - chunk_elements) / num_chunks : 1;
    if (stride == 0)
        stride = 1;

    for (std::size_t c = 0; c < num_chunks; ++c) {
        std::size_t chunk_start = c * stride;
        std::size_t chunk_len = std::min(chunk_elements, size - chunk_start);

        std::size_t chunk_sorted = 0;
        profile_chunk_simd(data + chunk_start, chunk_len, chunk_sorted, has_duplicates);

        sorted_sampled_pairs += chunk_sorted;
        total_sampled_pairs += (chunk_len > 0) ? (chunk_len - 1) : 0;

        // Check inter-chunk boundary if applicable
        if (c + 1 < num_chunks) {
            std::size_t next_start = (c + 1) * stride;
            if (chunk_start + chunk_len <= next_start && next_start < size) {
                if (data[chunk_start + chunk_len - 1] <= data[next_start]) {
                    sorted_sampled_pairs++;
                }
                if (data[chunk_start + chunk_len - 1] == data[next_start]) {
                    has_duplicates = true;
                }
                total_sampled_pairs++;
            }
        }
    }

    out_sortedness_ratio = (total_sampled_pairs > 0) ? static_cast<double>(sorted_sampled_pairs) /
                                                           static_cast<double>(total_sampled_pairs)
                                                     : 1.0;
    out_has_duplicates = has_duplicates;
}

} // namespace algoat::core::detail
