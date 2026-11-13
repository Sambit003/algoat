/**
 * @file vqsort.hpp
 * @brief Vectorized Quicksort (VQSort) implementation for AVX2 and AVX-512 architectures.
 *
 * Implements high-performance vectorized quicksort based on Wassenberg et al. (2022),
 * Bramas (2017), and Intel x86-simd-sort (2023). Features in-register bitonic sorting
 * networks for N <= 64, high-throughput SIMD two-way partitioning with in-place streaming
 * margins, adversarial invariant protection with sample-of-samples median-of-ninther,
 * and seamless fallback to PDQSort for non-primitive types and arbitrary comparators.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/meta/simd_sorting_network.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/pdqsort.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#if defined(__cpp_modules)
// In C++20 module interface compilation, hardware vector builtins trigger internal compiler
// errors across GCC 14 module state dependency tracking (PR114795). Delegate to scalar PDQSort
// during module BMI compilation.
#define ALGOAT_VQSORT_SCALAR_ONLY 1
#endif

#if !defined(ALGOAT_VQSORT_SCALAR_ONLY) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#endif

namespace algoat::sorting {

namespace detail::vqsort_detail {

/**
 * @brief Trait identifying arithmetic types supported by native vector sort paths.
 */
template <typename T>
inline constexpr bool is_simd_type_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

/**
 * @brief Evaluates whether standard less-than ordering is requested.
 */
template <typename T, typename Comp>
inline constexpr bool is_std_less_v =
    std::is_same_v<Comp, std::ranges::less> || std::is_same_v<Comp, std::less<>> ||
    std::is_same_v<Comp, std::less<T>>;

/**
 * @brief Evaluates whether standard greater-than ordering is requested.
 */
template <typename T, typename Comp>
inline constexpr bool is_std_greater_v =
    std::is_same_v<Comp, std::ranges::greater> || std::is_same_v<Comp, std::greater<>> ||
    std::is_same_v<Comp, std::greater<T>>;

/**
 * @brief Determines whether a sort configuration can use the hardware vector pipeline.
 */
#if defined(ALGOAT_VQSORT_SCALAR_ONLY)
template <typename T, typename Comp, typename Proj>
inline constexpr bool is_vqsort_applicable_v = false;
#else
template <typename T, typename Comp, typename Proj>
inline constexpr bool is_vqsort_applicable_v =
    is_simd_type_v<T> && (is_std_less_v<T, Comp> || is_std_greater_v<T, Comp>) &&
    std::is_same_v<Proj, std::identity>;
#endif

#if defined(__AVX2__) && !defined(ALGOAT_VQSORT_SCALAR_ONLY)

/**
 * @brief LUT entry holding complementary pack permutations for AVX2.
 */
template <std::size_t N> struct PermutePair {
    std::array<std::int32_t, N> lt;
    std::array<std::int32_t, N> ge;
};

/**
 * @brief Generates the 256-entry dual-packing LUT for AVX2 8-lane 32-bit partitioning.
 */
constexpr auto generate_avx2_lut32() {
    std::array<PermutePair<8>, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        std::array<std::int32_t, 8> lt_indices{};
        std::array<std::int32_t, 8> ge_indices{};
        int lt_count = 0;
        int ge_count = 0;
        for (int j = 0; j < 8; ++j) {
            if ((i >> j) & 1) {
                lt_indices[lt_count++] = j;
            } else {
                ge_indices[ge_count++] = j;
            }
        }
        lut[i] = {lt_indices, ge_indices};
    }
    return lut;
}

inline constexpr auto avx2_lut32 = generate_avx2_lut32();

/**
 * @brief Generates the 16-entry dual-packing LUT for AVX2 4-lane 64-bit partitioning.
 *
 * Each 64-bit element corresponds to two 32-bit dwords (2k, 2k+1) permuted across YMM.
 */
constexpr auto generate_avx2_lut64() {
    std::array<PermutePair<8>, 16> lut{};
    for (int i = 0; i < 16; ++i) {
        std::array<int, 4> lt_lanes{};
        std::array<int, 4> ge_lanes{};
        int lt_count = 0;
        int ge_count = 0;
        for (int j = 0; j < 4; ++j) {
            if ((i >> j) & 1) {
                lt_lanes[lt_count++] = j;
            } else {
                ge_lanes[ge_count++] = j;
            }
        }
        std::array<std::int32_t, 8> dword_lt{};
        for (int k = 0; k < lt_count; ++k) {
            int src = lt_lanes[k];
            dword_lt[2 * k] = 2 * src;
            dword_lt[2 * k + 1] = 2 * src + 1;
        }
        std::array<std::int32_t, 8> dword_ge{};
        for (int k = 0; k < ge_count; ++k) {
            int src = ge_lanes[k];
            dword_ge[2 * k] = 2 * src;
            dword_ge[2 * k + 1] = 2 * src + 1;
        }
        lut[i] = {dword_lt, dword_ge};
    }
    return lut;
}

inline constexpr auto avx2_lut64 = generate_avx2_lut64();

/**
 * @brief Precomputed mask tables for AVX2 32-bit masked stores (k = 0..8 valid elements).
 */
alignas(32) inline constexpr std::int32_t kStoreMask32Table[9][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},         {-1, 0, 0, 0, 0, 0, 0, 0},
    {-1, -1, 0, 0, 0, 0, 0, 0},       {-1, -1, -1, 0, 0, 0, 0, 0},
    {-1, -1, -1, -1, 0, 0, 0, 0},     {-1, -1, -1, -1, -1, 0, 0, 0},
    {-1, -1, -1, -1, -1, -1, 0, 0},   {-1, -1, -1, -1, -1, -1, -1, 0},
    {-1, -1, -1, -1, -1, -1, -1, -1},
};

/**
 * @brief Precomputed mask tables for AVX2 64-bit masked stores (k = 0..4 valid elements).
 */
alignas(32) inline constexpr std::int64_t kStoreMask64Table[5][4] = {
    {0, 0, 0, 0},          {-1LL, 0, 0, 0},          {-1LL, -1LL, 0, 0},
    {-1LL, -1LL, -1LL, 0}, {-1LL, -1LL, -1LL, -1LL},
};

/**
 * @brief Base case dispatcher for N <= 64 with zero heap allocation.
 */
template <typename T> void sort_base_network(T* data, std::size_t size) noexcept {
    algoat::meta::detail::simd_net::sort_small_simd(data, size);
}

#endif // __AVX2__

/**
 * @brief Drains center remainder and boundary cushions into left/right margins.
 *
 * Guaranteed to eliminate read-after-write hazards by buffering remainder data
 * on the stack before writing to target margin pointers.
 */
template <typename T, std::size_t VectorLanes>
inline std::size_t
drain_partition_buffers(T* data, std::size_t& left_write, std::size_t& right_write,
                        std::size_t read_left, std::size_t read_right,
                        const std::array<T, VectorLanes>& cushion_left,
                        const std::array<T, VectorLanes>& cushion_right, T pivot) noexcept {
    // 1. Buffer center remainder to prevent read-after-write hazard
    std::array<T, VectorLanes> rem_buf;
    std::size_t rem = read_right - read_left;
    for (std::size_t i = 0; i < rem; ++i) {
        rem_buf[i] = data[read_left + i];
    }

    for (std::size_t i = 0; i < rem; ++i) {
        if (rem_buf[i] < pivot) {
            data[left_write++] = rem_buf[i];
        } else {
            data[--right_write] = rem_buf[i];
        }
    }

    // 2. Drain cushions safely
    for (std::size_t i = 0; i < VectorLanes; ++i) {
        if (cushion_left[i] < pivot) {
            data[left_write++] = cushion_left[i];
        } else {
            data[--right_write] = cushion_left[i];
        }
    }
    for (std::size_t i = 0; i < VectorLanes; ++i) {
        if (cushion_right[i] < pivot) {
            data[left_write++] = cushion_right[i];
        } else {
            data[--right_write] = cushion_right[i];
        }
    }

    return left_write;
}

#if defined(__AVX2__) && !defined(ALGOAT_VQSORT_SCALAR_ONLY)

/**
 * @brief Two-way vectorized partitioning using AVX2 in-place dual stores and masked writes.
 */
template <typename T> std::size_t partition_avx2(T* data, std::size_t size, T pivot) noexcept {
    if constexpr (sizeof(T) == 4) {
        constexpr std::size_t VectorLanes = 8;
        if (size < 2 * VectorLanes) {
            auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
            return static_cast<std::size_t>(it - data);
        }

        alignas(32) std::array<T, VectorLanes> cushion_left;
        alignas(32) std::array<T, VectorLanes> cushion_right;
        for (std::size_t i = 0; i < VectorLanes; ++i) {
            cushion_left[i] = data[i];
            cushion_right[i] = data[size - VectorLanes + i];
        }

        std::size_t left_write = 0;
        std::size_t right_write = size;
        std::size_t read_left = VectorLanes;
        std::size_t read_right = size - VectorLanes;

        while (read_right >= read_left && (read_right - read_left) >= VectorLanes) {
            __m256i val;
            if ((read_left - left_write) <= (right_write - read_right)) {
                val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + read_left));
                read_left += VectorLanes;
            } else {
                read_right -= VectorLanes;
                val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + read_right));
            }

            int mask = 0;
            if constexpr (std::is_floating_point_v<T>) {
                __m256 v_p = _mm256_set1_ps(pivot);
                __m256 v_d = _mm256_castsi256_ps(val);
                __m256 cmp = _mm256_cmp_ps(v_d, v_p, _CMP_LT_OQ);
                mask = _mm256_movemask_ps(cmp);
            } else if constexpr (std::is_unsigned_v<T>) {
                __m256i v_p = _mm256_set1_epi32(static_cast<std::int32_t>(pivot));
                __m256i bias = _mm256_set1_epi32(static_cast<std::int32_t>(0x80000000u));
                __m256i biased_pivot = _mm256_xor_si256(v_p, bias);
                __m256i biased_val = _mm256_xor_si256(val, bias);
                __m256i cmp = _mm256_cmpgt_epi32(biased_pivot, biased_val);
                mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            } else {
                __m256i v_p = _mm256_set1_epi32(static_cast<std::int32_t>(pivot));
                __m256i cmp = _mm256_cmpgt_epi32(v_p, val);
                mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
            }

            int count_lt = _mm_popcnt_u32(static_cast<unsigned int>(mask));
            int count_ge = static_cast<int>(VectorLanes) - count_lt;

            __m256i perm_lt =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(avx2_lut32[mask].lt.data()));
            __m256i perm_ge =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(avx2_lut32[mask].ge.data()));
            __m256i packed_lt = _mm256_permutevar8x32_epi32(val, perm_lt);
            __m256i packed_ge = _mm256_permutevar8x32_epi32(val, perm_ge);

            __m256i mask_lt =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(kStoreMask32Table[count_lt]));
            __m256i mask_ge =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(kStoreMask32Table[count_ge]));

            // Exact masked stores strictly eliminate buffer overwrite hazards
            if constexpr (std::is_floating_point_v<T>) {
                _mm256_maskstore_ps(reinterpret_cast<float*>(data + left_write), mask_lt,
                                    _mm256_castsi256_ps(packed_lt));
                _mm256_maskstore_ps(reinterpret_cast<float*>(data + right_write - count_ge),
                                    mask_ge, _mm256_castsi256_ps(packed_ge));
            } else {
                _mm256_maskstore_epi32(reinterpret_cast<int*>(data + left_write), mask_lt,
                                       packed_lt);
                _mm256_maskstore_epi32(reinterpret_cast<int*>(data + right_write - count_ge),
                                       mask_ge, packed_ge);
            }

            left_write += count_lt;
            right_write -= count_ge;
        }

        return drain_partition_buffers<T, VectorLanes>(data, left_write, right_write, read_left,
                                                       read_right, cushion_left, cushion_right,
                                                       pivot);
    } else if constexpr (sizeof(T) == 8) {
        constexpr std::size_t VectorLanes = 4;
        if (size < 2 * VectorLanes) {
            auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
            return static_cast<std::size_t>(it - data);
        }

        alignas(32) std::array<T, VectorLanes> cushion_left;
        alignas(32) std::array<T, VectorLanes> cushion_right;
        for (std::size_t i = 0; i < VectorLanes; ++i) {
            cushion_left[i] = data[i];
            cushion_right[i] = data[size - VectorLanes + i];
        }

        std::size_t left_write = 0;
        std::size_t right_write = size;
        std::size_t read_left = VectorLanes;
        std::size_t read_right = size - VectorLanes;

        while (read_right >= read_left && (read_right - read_left) >= VectorLanes) {
            __m256i val;
            if ((read_left - left_write) <= (right_write - read_right)) {
                val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + read_left));
                read_left += VectorLanes;
            } else {
                read_right -= VectorLanes;
                val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + read_right));
            }

            int mask = 0;
            if constexpr (std::is_floating_point_v<T>) {
                __m256d v_p = _mm256_set1_pd(pivot);
                __m256d v_d = _mm256_castsi256_pd(val);
                __m256d cmp = _mm256_cmp_pd(v_d, v_p, _CMP_LT_OQ);
                mask = _mm256_movemask_pd(cmp);
            } else if constexpr (std::is_unsigned_v<T>) {
                __m256i v_p = _mm256_set1_epi64x(static_cast<std::int64_t>(pivot));
                __m256i bias = _mm256_set1_epi64x(static_cast<std::int64_t>(0x8000000000000000ULL));
                __m256i biased_pivot = _mm256_xor_si256(v_p, bias);
                __m256i biased_val = _mm256_xor_si256(val, bias);
                __m256i cmp = _mm256_cmpgt_epi64(biased_pivot, biased_val);
                mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp));
            } else {
                __m256i v_p = _mm256_set1_epi64x(static_cast<std::int64_t>(pivot));
                __m256i cmp = _mm256_cmpgt_epi64(v_p, val);
                mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp));
            }

            int count_lt = _mm_popcnt_u32(static_cast<unsigned int>(mask));
            int count_ge = static_cast<int>(VectorLanes) - count_lt;

            __m256i perm_lt =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(avx2_lut64[mask].lt.data()));
            __m256i perm_ge =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(avx2_lut64[mask].ge.data()));
            __m256i packed_lt = _mm256_permutevar8x32_epi32(val, perm_lt);
            __m256i packed_ge = _mm256_permutevar8x32_epi32(val, perm_ge);

            __m256i mask_lt =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(kStoreMask64Table[count_lt]));
            __m256i mask_ge =
                _mm256_load_si256(reinterpret_cast<const __m256i*>(kStoreMask64Table[count_ge]));

            if constexpr (std::is_floating_point_v<T>) {
                _mm256_maskstore_pd(reinterpret_cast<double*>(data + left_write), mask_lt,
                                    _mm256_castsi256_pd(packed_lt));
                _mm256_maskstore_pd(reinterpret_cast<double*>(data + right_write - count_ge),
                                    mask_ge, _mm256_castsi256_pd(packed_ge));
            } else {
                _mm256_maskstore_epi64(reinterpret_cast<long long*>(data + left_write), mask_lt,
                                       packed_lt);
                _mm256_maskstore_epi64(reinterpret_cast<long long*>(data + right_write - count_ge),
                                       mask_ge, packed_ge);
            }

            left_write += count_lt;
            right_write -= count_ge;
        }

        return drain_partition_buffers<T, VectorLanes>(data, left_write, right_write, read_left,
                                                       read_right, cushion_left, cushion_right,
                                                       pivot);
    } else {
        auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
        return static_cast<std::size_t>(it - data);
    }
}

#endif // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__) &&                      \
    defined(__AVX512BW__) && !defined(ALGOAT_VQSORT_SCALAR_ONLY)

/**
 * @brief Two-way vectorized partitioning using AVX-512 hardware compress-stores.
 */
template <typename T> std::size_t partition_avx512(T* data, std::size_t size, T pivot) noexcept {
    if constexpr (sizeof(T) == 4) {
        constexpr std::size_t VectorLanes = 16;
        if (size < 2 * VectorLanes) {
            auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
            return static_cast<std::size_t>(it - data);
        }

        alignas(64) std::array<T, VectorLanes> cushion_left;
        alignas(64) std::array<T, VectorLanes> cushion_right;
        for (std::size_t i = 0; i < VectorLanes; ++i) {
            cushion_left[i] = data[i];
            cushion_right[i] = data[size - VectorLanes + i];
        }

        std::size_t left_write = 0;
        std::size_t right_write = size;
        std::size_t read_left = VectorLanes;
        std::size_t read_right = size - VectorLanes;

        while (read_right >= read_left && (read_right - read_left) >= VectorLanes) {
            __mmask16 mask;
            if constexpr (std::is_floating_point_v<T>) {
                __m512 v_pivot = _mm512_set1_ps(pivot);
                __m512 val;
                if ((read_left - left_write) <= (right_write - read_right)) {
                    val = _mm512_loadu_ps(data + read_left);
                    read_left += VectorLanes;
                } else {
                    read_right -= VectorLanes;
                    val = _mm512_loadu_ps(data + read_right);
                }
                mask = _mm512_cmp_ps_mask(val, v_pivot, _CMP_LT_OQ);
                std::size_t count_lt = _mm_popcnt_u32(mask);
                std::size_t count_ge = VectorLanes - count_lt;

                _mm512_mask_compressstoreu_ps(data + left_write, mask, val);
                right_write -= count_ge;
                _mm512_mask_compressstoreu_ps(data + right_write, ~mask, val);
                left_write += count_lt;
            } else {
                __m512i v_pivot;
                if constexpr (std::is_unsigned_v<T>) {
                    v_pivot = _mm512_set1_epi32(static_cast<std::int32_t>(pivot));
                } else {
                    v_pivot = _mm512_set1_epi32(pivot);
                }

                __m512i val;
                if ((read_left - left_write) <= (right_write - read_right)) {
                    val = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + read_left));
                    read_left += VectorLanes;
                } else {
                    read_right -= VectorLanes;
                    val = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + read_right));
                }

                if constexpr (std::is_unsigned_v<T>) {
                    mask = _mm512_cmp_epu32_mask(val, v_pivot, _MM_CMPINT_LT);
                } else {
                    mask = _mm512_cmp_epi32_mask(val, v_pivot, _MM_CMPINT_LT);
                }

                std::size_t count_lt = _mm_popcnt_u32(mask);
                std::size_t count_ge = VectorLanes - count_lt;

                _mm512_mask_compressstoreu_epi32(data + left_write, mask, val);
                right_write -= count_ge;
                _mm512_mask_compressstoreu_epi32(data + right_write, ~mask, val);
                left_write += count_lt;
            }
        }

        return drain_partition_buffers<T, VectorLanes>(data, left_write, right_write, read_left,
                                                       read_right, cushion_left, cushion_right,
                                                       pivot);
    } else if constexpr (sizeof(T) == 8) {
        constexpr std::size_t VectorLanes = 8;
        if (size < 2 * VectorLanes) {
            auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
            return static_cast<std::size_t>(it - data);
        }

        alignas(64) std::array<T, VectorLanes> cushion_left;
        alignas(64) std::array<T, VectorLanes> cushion_right;
        for (std::size_t i = 0; i < VectorLanes; ++i) {
            cushion_left[i] = data[i];
            cushion_right[i] = data[size - VectorLanes + i];
        }

        std::size_t left_write = 0;
        std::size_t right_write = size;
        std::size_t read_left = VectorLanes;
        std::size_t read_right = size - VectorLanes;

        while (read_right >= read_left && (read_right - read_left) >= VectorLanes) {
            __mmask8 mask;
            if constexpr (std::is_floating_point_v<T>) {
                __m512d v_pivot = _mm512_set1_pd(pivot);
                __m512d val;
                if ((read_left - left_write) <= (right_write - read_right)) {
                    val = _mm512_loadu_pd(data + read_left);
                    read_left += VectorLanes;
                } else {
                    read_right -= VectorLanes;
                    val = _mm512_loadu_pd(data + read_right);
                }
                mask = _mm512_cmp_pd_mask(val, v_pivot, _CMP_LT_OQ);
                std::size_t count_lt = _mm_popcnt_u32(mask);
                std::size_t count_ge = VectorLanes - count_lt;

                _mm512_mask_compressstoreu_pd(data + left_write, mask, val);
                right_write -= count_ge;
                _mm512_mask_compressstoreu_pd(data + right_write, ~mask, val);
                left_write += count_lt;
            } else {
                __m512i v_pivot = _mm512_set1_epi64(static_cast<std::int64_t>(pivot));
                __m512i val;
                if ((read_left - left_write) <= (right_write - read_right)) {
                    val = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + read_left));
                    read_left += VectorLanes;
                } else {
                    read_right -= VectorLanes;
                    val = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + read_right));
                }

                if constexpr (std::is_unsigned_v<T>) {
                    mask = _mm512_cmp_epu64_mask(val, v_pivot, _MM_CMPINT_LT);
                } else {
                    mask = _mm512_cmp_epi64_mask(val, v_pivot, _MM_CMPINT_LT);
                }

                std::size_t count_lt = _mm_popcnt_u32(mask);
                std::size_t count_ge = VectorLanes - count_lt;

                _mm512_mask_compressstoreu_epi64(data + left_write, mask, val);
                right_write -= count_ge;
                _mm512_mask_compressstoreu_epi64(data + right_write, ~mask, val);
                left_write += count_lt;
            }
        }

        return drain_partition_buffers<T, VectorLanes>(data, left_write, right_write, read_left,
                                                       read_right, cushion_left, cushion_right,
                                                       pivot);
    } else {
        auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
        return static_cast<std::size_t>(it - data);
    }
}

#endif // AVX512

/**
 * @brief Computes median of three values.
 */
template <typename T> constexpr T median3(T a, T b, T c) noexcept {
    if (a < b) {
        if (b < c) {
            return b;
        }
        return (a < c) ? c : a;
    } else {
        if (a < c) {
            return a;
        }
        return (b < c) ? c : b;
    }
}

/**
 * @brief Evaluates Tukey's ninther (pseudo-median of 9 elements).
 */
template <typename T>
constexpr T ninther(const T* data, std::size_t offset, std::size_t stride) noexcept {
    T m1 = median3(data[offset], data[offset + stride], data[offset + 2 * stride]);
    T m2 = median3(data[offset + 3 * stride], data[offset + 4 * stride], data[offset + 5 * stride]);
    T m3 = median3(data[offset + 6 * stride], data[offset + 7 * stride], data[offset + 8 * stride]);
    return median3(m1, m2, m3);
}

/**
 * @brief Selects a robust pivot using true sample-of-samples median-of-ninther.
 *
 * For N >= 1024, samples 81 elements via 9 sub-ninthers with PivotRank protection
 * against adversarial worst cases. For intermediate sizes, uses Tukey's ninther or median-of-3.
 */
template <typename T> T choose_pivot_smart(const T* data, std::size_t size) noexcept {
    if (size >= 1024) {
        std::size_t block_step = size / 9;
        std::size_t stride = block_step / 9;
        if (stride == 0) {
            stride = 1;
        }

        std::array<T, 9> ninthers;
        for (std::size_t i = 0; i < 9; ++i) {
            ninthers[i] = ninther(data, i * block_step, stride);
        }

        std::sort(ninthers.begin(), ninthers.end());

        // PivotRank protection: ensure pivot < max(samples) to prevent O(N^2) degradation
        T max_sample = ninthers.back();
        for (int i = 4; i >= 0; --i) {
            if (ninthers[i] < max_sample) {
                return ninthers[i];
            }
        }
        return ninthers[0];
    } else if (size >= 128) {
        std::size_t stride = size / 9;
        return ninther(data, 0, stride);
    } else {
        return median3(data[0], data[size / 2], data[size - 1]);
    }
}

/**
 * @brief Recursive VQSort engine with hardware vector partitioning and Heapsort fallback.
 */
template <typename T>
void vqsort_recursive(T* data, std::size_t size, std::size_t depth_limit) noexcept {
    while (size > 64) {
        if (depth_limit == 0) {
            algoat::sorting::heapsort(std::span<T>(data, size));
            return;
        }
        --depth_limit;

        T pivot = choose_pivot_smart(data, size);
        std::size_t mid;

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__) &&                      \
    defined(__AVX512BW__) && !defined(ALGOAT_VQSORT_SCALAR_ONLY)
        mid = partition_avx512(data, size, pivot);
#elif defined(__AVX2__)
        mid = partition_avx2(data, size, pivot);
#else
        auto it = std::partition(data, data + size, [pivot](const T& x) { return x < pivot; });
        mid = static_cast<std::size_t>(it - data);
#endif

        if (mid == 0) {
            auto it = std::partition(data, data + size, [pivot](const T& x) { return x == pivot; });
            std::size_t eq_count = static_cast<std::size_t>(it - data);
            if (eq_count == size) {
                return;
            }
            data += eq_count;
            size -= eq_count;
            continue;
        } else if (mid == size) {
            --size;
            continue;
        }

        if (mid < size - mid) {
            vqsort_recursive(data, mid, depth_limit);
            data += mid;
            size -= mid;
        } else {
            vqsort_recursive(data + mid, size - mid, depth_limit);
            size = mid;
        }
    }

    // Base case: in-register sorting network
#if defined(__AVX2__) && !defined(ALGOAT_VQSORT_SCALAR_ONLY)
    sort_base_network(data, size);
#else
    std::sort(data, data + size);
#endif
}

} // namespace detail::vqsort_detail

// ============================================================================
// Public Interfaces
// ============================================================================

/**
 * @brief Vectorized quicksort over an iterator-sentinel range.
 *
 * Automatically dispatches to AVX2 or AVX-512 vector pipelines for supported
 * scalar types with standard comparators, and cleanly delegates to PDQSort
 * for non-primitive types and arbitrary comparator/projection combinations.
 *
 * @tparam I Random-access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable applied to elements before comparison.
 * @param first Beginning of the range to sort.
 * @param last Sentinel marking the end of the range.
 * @param comp Comparator callable (default: std::ranges::less).
 * @param proj Projection callable (default: std::identity).
 * @return Iterator pointing to the end of the sorted range.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr I vqsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    using T = std::iter_value_t<I>;
    auto num_elements = std::ranges::distance(first, last);
    if (num_elements <= 1) {
        return first + num_elements;
    }

    if constexpr (std::contiguous_iterator<I> &&
                  detail::vqsort_detail::is_vqsort_applicable_v<T, Comp, Proj>) {
        if (!std::is_constant_evaluated()) {
#if (defined(__AVX2__) || defined(__AVX512F__)) && !defined(ALGOAT_VQSORT_SCALAR_ONLY)
            T* ptr = std::to_address(first);
            std::size_t size = static_cast<std::size_t>(num_elements);
            std::size_t depth_limit = 2 * std::bit_width(size) + 4;

            detail::vqsort_detail::vqsort_recursive(ptr, size, depth_limit);

            if constexpr (detail::vqsort_detail::is_std_greater_v<T, Comp>) {
                std::reverse(ptr, ptr + size);
            }
            return first + num_elements;
#endif
        }
    }

    // Clean zero-overhead fallback to branchless pdqsort
    algoat::sorting::pdqsort(first, last, std::move(comp), std::move(proj));
    return first + num_elements;
}

/**
 * @brief Vectorized quicksort over a C++20 random access range.
 *
 * @tparam R Random-access range type.
 * @tparam Comp Strict weak ordering comparator callable.
 * @tparam Proj Projection callable.
 * @param data Range of elements to sort in-place.
 * @param comp Comparator callable.
 * @param proj Projection callable.
 * @return Borrowed iterator pointing to the end of the sorted range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> vqsort(R&& data, Comp comp = {}, Proj proj = {}) {
    return vqsort(std::ranges::begin(data), std::ranges::end(data), std::move(comp),
                  std::move(proj));
}

/**
 * @brief In-place vectorized quicksort over a contiguous span with custom comparator.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator type.
 * @param data Contiguous span of elements.
 * @param comp Comparator callable.
 */
template <typename T, typename Compare> void vqsort(std::span<T> data, Compare comp) {
    vqsort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @brief In-place vectorized quicksort over a contiguous span with default ascending order.
 *
 * @tparam T Element type.
 * @param data Contiguous span of elements.
 */
template <typename T> void vqsort(std::span<T> data) {
    vqsort(data.begin(), data.end(), std::ranges::less{}, std::identity{});
}

/**
 * @struct VQSort
 * @brief Zero-overhead adapter struct for Vectorized Quicksort (VQSort).
 *
 * Fully conforms to the @c algoat::sorting::SortAlgorithm concept for static polymorphism
 * and decentralized registry storage.
 */
struct VQSort {
    /**
     * @brief Unique algorithm identifier name.
     * @return "vqsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "vqsort";
    }

    /**
     * @brief Preferred minimum dataset threshold where SIMD partitioning becomes effective.
     * @return 16
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 16;
    }

    /**
     * @brief Sorts the span in-place using default comparator.
     */
    template <typename T> void sort(std::span<T> data) const {
        vqsort(data);
    }

    /**
     * @brief Sorts the span in-place using a custom comparator.
     */
    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        vqsort(data, std::move(comp));
    }
};

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "vqsort", ::algoat::sorting::VQSort)
