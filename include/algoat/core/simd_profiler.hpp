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

struct ChunkProfileResult {
    std::size_t sorted_pairs{0};
    bool has_duplicates{false};
};

/**
 * @brief Evaluates adjacent sortedness and equality across a contiguous memory chunk using SIMD.
 */
template <typename T>
HWY_INLINE ChunkProfileResult profile_chunk_simd(const T* ptr, std::size_t len) {
    ChunkProfileResult res;
    if (len <= 1)
        return res;

    const hn::ScalableTag<T> d;
    const std::size_t N = hn::Lanes(d);

    std::size_t i = 0;
    // Process full SIMD vector widths
    if (N > 0) {
        for (; i + N < len; i += N) {
            auto v1 = hn::LoadU(d, ptr + i);
            auto v2 = hn::LoadU(d, ptr + i + 1);

            auto le_mask = hn::Le(v1, v2);
            auto eq_mask = hn::Eq(v1, v2);

            res.sorted_pairs += hn::CountTrue(d, le_mask);
            if (hn::CountTrue(d, eq_mask) > 0) {
                res.has_duplicates = true;
            }
        }
    }

    // Process remaining scalar tail in chunk
    for (; i + 1 < len; ++i) {
        if (ptr[i] <= ptr[i + 1]) {
            res.sorted_pairs++;
        }
        if (ptr[i] == ptr[i + 1]) {
            res.has_duplicates = true;
        }
    }

    return res;
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

        ChunkProfileResult chunk_res = profile_chunk_simd(data + chunk_start, chunk_len);

        sorted_sampled_pairs += chunk_res.sorted_pairs;
        if (chunk_res.has_duplicates) {
            has_duplicates = true;
        }
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
