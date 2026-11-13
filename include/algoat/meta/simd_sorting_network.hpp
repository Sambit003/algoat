/**
 * @file simd_sorting_network.hpp
 * @brief In-register SIMD sorting networks for fixed-size spans (N <= 64).
 *
 * Implements zero-spill in-register Batcher bitonic and optimal column-wise sorting
 * networks for N in {8, 16, 32, 64} across int32_t, uint32_t, int64_t, uint64_t, float,
 * and double using AVX-512 and AVX2 intrinsics, with a noexcept scalar fallback.
 */

#pragma once

#include "algoat/meta/sorting_network.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#if !defined(ALGOAT_SIMD_SORTING_NETWORK_SCALAR_ONLY) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ALGOAT_SIMD_NET_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALGOAT_SIMD_NET_INLINE __forceinline
#else
#define ALGOAT_SIMD_NET_INLINE inline
#endif

namespace algoat::meta {

namespace detail::simd_net {

/**
 * @brief Canonical SIMD element key mapping integral aliases (e.g. long vs long long)
 * to fixed-width standard types while preserving float/double.
 */
template <typename T> struct CanonicalSimdType {
    using type = void;
};

template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool> && sizeof(T) == 4)
struct CanonicalSimdType<T> {
    using type = std::conditional_t<std::is_signed_v<T>, std::int32_t, std::uint32_t>;
};

template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool> && sizeof(T) == 8)
struct CanonicalSimdType<T> {
    using type = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
};

template <> struct CanonicalSimdType<float> {
    using type = float;
};

template <> struct CanonicalSimdType<double> {
    using type = double;
};

template <typename T>
using canonical_simd_t = typename CanonicalSimdType<std::remove_cv_t<T>>::type;

template <typename T>
inline constexpr bool is_simd_network_type_v = !std::is_void_v<canonical_simd_t<T>>;

/**
 * @brief Noexcept scalar fallback for non-SIMD arithmetic types and module BMI builds.
 */
template <std::size_t N, typename T>
ALGOAT_SIMD_NET_INLINE void scalar_fallback_sort(std::span<T, N> data) noexcept {
    if constexpr (N <= 16) {
        algoat::meta::sort_network<T, N>(data);
    } else {
        for (std::size_t i = 1; i < N; ++i) {
            T key = data[i];
            std::size_t j = i;
            while (j > 0 && key < data[j - 1]) {
                data[j] = data[j - 1];
                --j;
            }
            data[j] = key;
        }
    }
}

#if (defined(__AVX2__) || defined(__AVX512F__)) && !defined(ALGOAT_SIMD_SORTING_NETWORK_SCALAR_ONLY)

/**
 * @brief Generic column-wise optimal network + bitonic merge across M vector registers.
 */
template <typename Ops, std::size_t M> struct MultiVecNetwork {
    using Vec = typename Ops::Vec;

    ALGOAT_SIMD_NET_INLINE static void coex(Vec& a, Vec& b) noexcept {
        // Reversed operand order in max(b, a) preserves -0.0 / +0.0 sign bits on x86 FP units.
        Vec lo = Ops::min(a, b);
        Vec hi = Ops::max(b, a);
        a = lo;
        b = hi;
    }

    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_vec(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return Ops::template swap_n<2>(v);
        } else {
            return Ops::template reverse_n<Scale>(v);
        }
    }

    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static void merge_substep_one(Vec& a, Vec& b) noexcept {
        Vec rev = reverse_vec<Scale>(b);
        coex(a, rev);
        b = rev;
    }

    template <std::size_t GroupSize, std::size_t Scale, std::size_t... Is>
    ALGOAT_SIMD_NET_INLINE static void merge_substep_pairs(Vec* regs,
                                                           std::index_sequence<Is...>) noexcept {
        (merge_substep_one<Scale>(regs[Is], regs[GroupSize - 1 - Is]), ...);
    }

    template <std::size_t GroupSize, std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static void merge_substep(Vec* regs) noexcept {
        if constexpr (GroupSize >= 2) {
            constexpr std::size_t Half = GroupSize / 2;
            merge_substep_pairs<GroupSize, Scale>(regs, std::make_index_sequence<Half>{});
            merge_substep<Half, Scale>(regs);
            merge_substep<Half, Scale>(regs + Half);
        }
    }

    template <std::size_t Scale, std::size_t SubScale>
    ALGOAT_SIMD_NET_INLINE static void internal_merge_tail(Vec& v) noexcept {
        if constexpr (SubScale >= 2) {
            Vec swapped = Ops::template swap_n<SubScale>(v);
            // Same operand order (v, swapped) ensures lo picks swapped[i]=v[j] and hi picks
            // swapped[j]=v[i].
            Vec lo = Ops::min(v, swapped);
            Vec hi = Ops::max(v, swapped);
            v = Ops::template merge_n<SubScale>(lo, hi);
            internal_merge_tail<Scale, SubScale / 2>(v);
        }
    }

    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static void internal_merge_one(Vec& v) noexcept {
        Vec rev = reverse_vec<Scale>(v);
        Vec lo = Ops::min(v, rev);
        Vec hi = Ops::max(v, rev);
        v = Ops::template merge_n<Scale>(lo, hi);
        internal_merge_tail<Scale, Scale / 2>(v);
    }

    template <std::size_t Scale, std::size_t... Is>
    ALGOAT_SIMD_NET_INLINE static void internal_merge_all(Vec* regs,
                                                          std::index_sequence<Is...>) noexcept {
        (internal_merge_one<Scale>(regs[Is]), ...);
    }

    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static void merge_all_scales(Vec* regs) noexcept {
        if constexpr (Scale <= Ops::lanes) {
            merge_substep<M, Scale>(regs);
            internal_merge_all<Scale>(regs, std::make_index_sequence<M>{});
            merge_all_scales<Scale * 2>(regs);
        }
    }

    template <std::size_t... Is>
    ALGOAT_SIMD_NET_INLINE static void apply_network_pairs(Vec* regs,
                                                           std::index_sequence<Is...>) noexcept {
        constexpr auto net = algoat::meta::network_table_v<M>;
        (coex(regs[net.pairs[Is].i], regs[net.pairs[Is].j]), ...);
    }

    ALGOAT_SIMD_NET_INLINE static void sort_regs(Vec* regs) noexcept {
        if constexpr (M >= 2) {
            constexpr auto net = algoat::meta::network_table_v<M>;
            apply_network_pairs(regs, std::make_index_sequence<net.count>{});
        }
        merge_all_scales<2>(regs);
    }

    template <typename T, std::size_t... Is>
    ALGOAT_SIMD_NET_INLINE static void load_all(const T* ptr, Vec* regs,
                                                std::index_sequence<Is...>) noexcept {
        ((regs[Is] = Ops::load(ptr + Is * Ops::lanes)), ...);
    }

    template <typename T, std::size_t... Is>
    ALGOAT_SIMD_NET_INLINE static void store_all(T* ptr, const Vec* regs,
                                                 std::index_sequence<Is...>) noexcept {
        (Ops::store(ptr + Is * Ops::lanes, regs[Is]), ...);
    }

    template <typename T> ALGOAT_SIMD_NET_INLINE static void sort_memory(T* ptr) noexcept {
        Vec regs[M];
        load_all(ptr, regs, std::make_index_sequence<M>{});
        sort_regs(regs);
        store_all(ptr, regs, std::make_index_sequence<M>{});
    }
};

#endif // (__AVX2__ || __AVX512F__) && !SCALAR_ONLY

#if defined(__AVX2__) && !defined(ALGOAT_SIMD_SORTING_NETWORK_SCALAR_ONLY)

template <typename T> struct Avx2Ops32;

template <> struct Avx2Ops32<std::int32_t> {
    using Vec = __m256i;
    static constexpr std::size_t lanes = 8;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm256_loadu_si256(static_cast<const __m256i*>(ptr));
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm256_storeu_si256(static_cast<__m256i*>(ptr), v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm256_min_epi32(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm256_max_epi32(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_shuffle_epi32(v, 0xB1);
        } else if constexpr (Scale == 4) {
            return _mm256_shuffle_epi32(v, 0x4E);
        } else {
            static_assert(Scale == 8);
            return _mm256_permute2x128_si256(v, v, 0x01);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        if constexpr (Scale == 4) {
            return _mm256_shuffle_epi32(v, 0x1B);
        } else {
            static_assert(Scale == 8);
            Vec r4 = _mm256_shuffle_epi32(v, 0x1B);
            return _mm256_permute2x128_si256(r4, r4, 0x01);
        }
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_blend_epi32(lo, hi, 0xAA);
        } else if constexpr (Scale == 4) {
            return _mm256_blend_epi32(lo, hi, 0xCC);
        } else {
            static_assert(Scale == 8);
            return _mm256_blend_epi32(lo, hi, 0xF0);
        }
    }
};

template <> struct Avx2Ops32<std::uint32_t> : Avx2Ops32<std::int32_t> {
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm256_min_epu32(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm256_max_epu32(a, b);
    }
};

template <> struct Avx2Ops32<float> {
    using Vec = __m256;
    static constexpr std::size_t lanes = 8;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm256_loadu_ps(static_cast<const float*>(ptr));
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm256_storeu_ps(static_cast<float*>(ptr), v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm256_min_ps(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm256_max_ps(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_permute_ps(v, 0xB1);
        } else if constexpr (Scale == 4) {
            return _mm256_permute_ps(v, 0x4E);
        } else {
            static_assert(Scale == 8);
            return _mm256_permute2f128_ps(v, v, 0x01);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        if constexpr (Scale == 4) {
            return _mm256_permute_ps(v, 0x1B);
        } else {
            static_assert(Scale == 8);
            Vec r4 = _mm256_permute_ps(v, 0x1B);
            return _mm256_permute2f128_ps(r4, r4, 0x01);
        }
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_blend_ps(lo, hi, 0xAA);
        } else if constexpr (Scale == 4) {
            return _mm256_blend_ps(lo, hi, 0xCC);
        } else {
            static_assert(Scale == 8);
            return _mm256_blend_ps(lo, hi, 0xF0);
        }
    }
};

template <typename T> struct Avx2Ops64;

template <> struct Avx2Ops64<std::int64_t> {
    using Vec = __m256i;
    static constexpr std::size_t lanes = 4;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm256_loadu_si256(static_cast<const __m256i*>(ptr));
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm256_storeu_si256(static_cast<__m256i*>(ptr), v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        Vec gt = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(a, b, gt);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        Vec gt = _mm256_cmpgt_epi64(a, b);
        return _mm256_blendv_epi8(b, a, gt);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_shuffle_epi32(v, 0x4E);
        } else {
            static_assert(Scale == 4);
            return _mm256_permute2x128_si256(v, v, 0x01);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        static_assert(Scale == 4);
        return _mm256_permute4x64_epi64(v, 0x1B);
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_blend_epi32(lo, hi, 0xCC);
        } else {
            static_assert(Scale == 4);
            return _mm256_blend_epi32(lo, hi, 0xF0);
        }
    }
};

template <> struct Avx2Ops64<double> {
    using Vec = __m256d;
    static constexpr std::size_t lanes = 4;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm256_loadu_pd(static_cast<const double*>(ptr));
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm256_storeu_pd(static_cast<double*>(ptr), v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm256_min_pd(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm256_max_pd(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_permute_pd(v, 0x05);
        } else {
            static_assert(Scale == 4);
            return _mm256_permute2f128_pd(v, v, 0x01);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        static_assert(Scale == 4);
        return _mm256_permute4x64_pd(v, 0x1B);
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm256_blend_pd(lo, hi, 0xA);
        } else {
            static_assert(Scale == 4);
            return _mm256_blend_pd(lo, hi, 0xC);
        }
    }
};

ALGOAT_SIMD_NET_INLINE __m256i avx2_sign_bias_u64() noexcept {
    // Re-materialized in load/store helpers so bias is not live across sort_regs().
    return _mm256_slli_epi64(_mm256_set1_epi32(-1), 63);
}

template <typename T, std::size_t... Is>
ALGOAT_SIMD_NET_INLINE void load_avx2_uint64_biased(const T* ptr, __m256i* regs,
                                                    std::index_sequence<Is...>) noexcept {
    const __m256i bias = avx2_sign_bias_u64();
    ((regs[Is] = _mm256_xor_si256(Avx2Ops64<std::int64_t>::load(ptr + Is * 4), bias)), ...);
}

template <typename T, std::size_t... Is>
ALGOAT_SIMD_NET_INLINE void store_avx2_uint64_biased(T* ptr, const __m256i* regs,
                                                     std::index_sequence<Is...>) noexcept {
    const __m256i bias = avx2_sign_bias_u64();
    (Avx2Ops64<std::int64_t>::store(ptr + Is * 4, _mm256_xor_si256(regs[Is], bias)), ...);
}

template <std::size_t M, typename T>
ALGOAT_SIMD_NET_INLINE void sort_avx2_uint64_biased(T* ptr) noexcept {
    __m256i regs[M];
    load_avx2_uint64_biased(ptr, regs, std::make_index_sequence<M>{});
    MultiVecNetwork<Avx2Ops64<std::int64_t>, M>::sort_regs(regs);
    store_avx2_uint64_biased(ptr, regs, std::make_index_sequence<M>{});
}

#endif // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__) &&                      \
    defined(__AVX512BW__) && !defined(ALGOAT_SIMD_SORTING_NETWORK_SCALAR_ONLY)

template <typename T> struct Avx512Ops32;

template <> struct Avx512Ops32<std::int32_t> {
    using Vec = __m512i;
    static constexpr std::size_t lanes = 16;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm512_loadu_si512(ptr);
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm512_storeu_si512(ptr, v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm512_min_epi32(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm512_max_epi32(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_shuffle_epi32(v, _MM_PERM_CDAB);
        } else if constexpr (Scale == 4) {
            return _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        } else if constexpr (Scale == 8) {
            return _mm512_shuffle_i64x2(v, v, 0xB1);
        } else {
            static_assert(Scale == 16);
            return _mm512_shuffle_i64x2(v, v, 0x4E);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        if constexpr (Scale == 4) {
            return _mm512_shuffle_epi32(v, _MM_PERM_ABCD);
        } else if constexpr (Scale == 8) {
            Vec r4 = _mm512_shuffle_epi32(v, _MM_PERM_ABCD);
            return _mm512_shuffle_i64x2(r4, r4, 0xB1);
        } else {
            static_assert(Scale == 16);
            Vec r4 = _mm512_shuffle_epi32(v, _MM_PERM_ABCD);
            return _mm512_shuffle_i64x2(r4, r4, 0x1B);
        }
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_mask_blend_epi32(0xAAAA, lo, hi);
        } else if constexpr (Scale == 4) {
            return _mm512_mask_blend_epi32(0xCCCC, lo, hi);
        } else if constexpr (Scale == 8) {
            return _mm512_mask_blend_epi32(0xF0F0, lo, hi);
        } else {
            static_assert(Scale == 16);
            return _mm512_mask_blend_epi32(0xFF00, lo, hi);
        }
    }
};

template <> struct Avx512Ops32<std::uint32_t> : Avx512Ops32<std::int32_t> {
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm512_min_epu32(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm512_max_epu32(a, b);
    }
};

template <> struct Avx512Ops32<float> {
    using Vec = __m512;
    static constexpr std::size_t lanes = 16;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm512_loadu_ps(ptr);
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm512_storeu_ps(ptr, v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm512_min_ps(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm512_max_ps(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_permute_ps(v, 0xB1);
        } else if constexpr (Scale == 4) {
            return _mm512_permute_ps(v, 0x4E);
        } else if constexpr (Scale == 8) {
            return _mm512_shuffle_f32x4(v, v, 0xB1);
        } else {
            static_assert(Scale == 16);
            return _mm512_shuffle_f32x4(v, v, 0x4E);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        if constexpr (Scale == 4) {
            return _mm512_permute_ps(v, 0x1B);
        } else if constexpr (Scale == 8) {
            Vec r4 = _mm512_permute_ps(v, 0x1B);
            return _mm512_shuffle_f32x4(r4, r4, 0xB1);
        } else {
            static_assert(Scale == 16);
            Vec r4 = _mm512_permute_ps(v, 0x1B);
            return _mm512_shuffle_f32x4(r4, r4, 0x1B);
        }
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_mask_blend_ps(0xAAAA, lo, hi);
        } else if constexpr (Scale == 4) {
            return _mm512_mask_blend_ps(0xCCCC, lo, hi);
        } else if constexpr (Scale == 8) {
            return _mm512_mask_blend_ps(0xF0F0, lo, hi);
        } else {
            static_assert(Scale == 16);
            return _mm512_mask_blend_ps(0xFF00, lo, hi);
        }
    }
};

template <typename T> struct Avx512Ops64;

template <> struct Avx512Ops64<std::int64_t> {
    using Vec = __m512i;
    static constexpr std::size_t lanes = 8;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm512_loadu_si512(ptr);
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm512_storeu_si512(ptr, v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm512_min_epi64(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm512_max_epi64(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_shuffle_epi32(v, _MM_PERM_BADC);
        } else if constexpr (Scale == 4) {
            return _mm512_shuffle_i64x2(v, v, 0xB1);
        } else {
            static_assert(Scale == 8);
            return _mm512_shuffle_i64x2(v, v, 0x4E);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        if constexpr (Scale == 4) {
            return _mm512_permutex_epi64(v, 0x1B);
        } else {
            static_assert(Scale == 8);
            Vec r2 = _mm512_shuffle_epi32(v, _MM_PERM_BADC);
            return _mm512_shuffle_i64x2(r2, r2, 0x1B);
        }
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_mask_blend_epi64(0xAA, lo, hi);
        } else if constexpr (Scale == 4) {
            return _mm512_mask_blend_epi64(0xCC, lo, hi);
        } else {
            static_assert(Scale == 8);
            return _mm512_mask_blend_epi64(0xF0, lo, hi);
        }
    }
};

template <> struct Avx512Ops64<std::uint64_t> : Avx512Ops64<std::int64_t> {
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm512_min_epu64(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm512_max_epu64(a, b);
    }
};

template <> struct Avx512Ops64<double> {
    using Vec = __m512d;
    static constexpr std::size_t lanes = 8;

    ALGOAT_SIMD_NET_INLINE static Vec load(const void* ptr) noexcept {
        return _mm512_loadu_pd(ptr);
    }
    ALGOAT_SIMD_NET_INLINE static void store(void* ptr, Vec v) noexcept {
        _mm512_storeu_pd(ptr, v);
    }
    ALGOAT_SIMD_NET_INLINE static Vec min(Vec a, Vec b) noexcept {
        return _mm512_min_pd(a, b);
    }
    ALGOAT_SIMD_NET_INLINE static Vec max(Vec a, Vec b) noexcept {
        return _mm512_max_pd(a, b);
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec swap_n(Vec v) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_permute_pd(v, 0x55);
        } else if constexpr (Scale == 4) {
            return _mm512_shuffle_f64x2(v, v, 0xB1);
        } else {
            static_assert(Scale == 8);
            return _mm512_shuffle_f64x2(v, v, 0x4E);
        }
    }
    template <std::size_t Scale> ALGOAT_SIMD_NET_INLINE static Vec reverse_n(Vec v) noexcept {
        if constexpr (Scale == 4) {
            return _mm512_permutex_pd(v, 0x1B);
        } else {
            static_assert(Scale == 8);
            Vec r2 = _mm512_permute_pd(v, 0x55);
            return _mm512_shuffle_f64x2(r2, r2, 0x1B);
        }
    }
    template <std::size_t Scale>
    ALGOAT_SIMD_NET_INLINE static Vec merge_n(Vec lo, Vec hi) noexcept {
        if constexpr (Scale == 2) {
            return _mm512_mask_blend_pd(0xAA, lo, hi);
        } else if constexpr (Scale == 4) {
            return _mm512_mask_blend_pd(0xCC, lo, hi);
        } else {
            static_assert(Scale == 8);
            return _mm512_mask_blend_pd(0xF0, lo, hi);
        }
    }
};

#endif // AVX512

template <std::size_t N, typename T>
ALGOAT_SIMD_NET_INLINE void dispatch_simd_network(T* ptr) noexcept {
    using K = canonical_simd_t<T>;

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__) &&                      \
    defined(__AVX512BW__) && !defined(ALGOAT_SIMD_SORTING_NETWORK_SCALAR_ONLY)
    if constexpr (sizeof(K) == 4) {
        if constexpr (N == 8) {
            MultiVecNetwork<Avx2Ops32<K>, 1>::sort_memory(ptr);
        } else {
            MultiVecNetwork<Avx512Ops32<K>, N / 16>::sort_memory(ptr);
        }
    } else {
        MultiVecNetwork<Avx512Ops64<K>, N / 8>::sort_memory(ptr);
    }
#elif defined(__AVX2__) && !defined(ALGOAT_SIMD_SORTING_NETWORK_SCALAR_ONLY)
    if constexpr (sizeof(K) == 4) {
        MultiVecNetwork<Avx2Ops32<K>, N / 8>::sort_memory(ptr);
    } else if constexpr (std::is_same_v<K, std::uint64_t>) {
        sort_avx2_uint64_biased<N / 4>(ptr);
    } else {
        MultiVecNetwork<Avx2Ops64<K>, N / 4>::sort_memory(ptr);
    }
#else
    scalar_fallback_sort<N, T>(std::span<T, N>{ptr, N});
#endif
}

template <std::size_t N, typename T>
ALGOAT_SIMD_NET_INLINE void pad_and_sort(T* data, std::size_t size) noexcept {
    alignas(64) std::array<T, N> buffer;
    for (std::size_t i = 0; i < size; ++i) {
        buffer[i] = data[i];
    }
    T sentinel;
    if constexpr (std::is_floating_point_v<T>) {
        sentinel = std::numeric_limits<T>::infinity();
    } else {
        sentinel = std::numeric_limits<T>::max();
    }
    for (std::size_t i = size; i < N; ++i) {
        buffer[i] = sentinel;
    }
    dispatch_simd_network<N>(buffer.data());
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = buffer[i];
    }
}

/**
 * @brief Zero-allocation SIMD sorting network base-case driver for arbitrary N <= 64.
 */
template <typename T> inline void sort_small_simd(T* data, std::size_t size) noexcept {
    if (size <= 1) {
        return;
    }
    if (size == 2) {
        if (data[1] < data[0]) {
            std::swap(data[0], data[1]);
        }
        return;
    }
    if constexpr (is_simd_network_type_v<T>) {
        if (size <= 8) {
            pad_and_sort<8>(data, size);
            return;
        }
        if (size <= 16) {
            pad_and_sort<16>(data, size);
            return;
        }
        if (size <= 32) {
            pad_and_sort<32>(data, size);
            return;
        }
        if (size <= 64) {
            pad_and_sort<64>(data, size);
            return;
        }
    }
    std::sort(data, data + size);
}

} // namespace detail::simd_net

/**
 * @struct SIMDSortingNetwork
 * @brief In-register SIMD sorting network engine for fixed-size spans (N in {8, 16, 32, 64}).
 *
 * Uses AVX-512 / AVX2 compare-exchange and immediate shuffle networks for int32_t, uint32_t,
 * int64_t, uint64_t, float, and double, and falls back to a noexcept scalar sorting network
 * for all other arithmetic types.
 *
 * @tparam N Fixed span extent (8, 16, 32, or 64).
 * @tparam T Arithmetic element type.
 */
template <std::size_t N, typename T>
    requires(N == 8 || N == 16 || N == 32 || N == 64) && std::is_arithmetic_v<T>
struct SIMDSortingNetwork {
    /**
     * @brief Sorts a fixed-size span in-place inside SIMD vector registers.
     * @param data Fixed-extent span of N elements to sort.
     */
    static void sort(std::span<T, N> data) noexcept {
        if constexpr (detail::simd_net::is_simd_network_type_v<T>) {
            detail::simd_net::dispatch_simd_network<N>(data.data());
        } else {
            detail::simd_net::scalar_fallback_sort<N, T>(data);
        }
    }
};

/**
 * @brief Convenience free function to sort a fixed-extent span using SIMDSortingNetwork.
 *
 * @tparam N Fixed span extent (8, 16, 32, or 64).
 * @tparam T Arithmetic element type.
 * @param data Fixed-extent span of N elements to sort in-place.
 */
template <std::size_t N, typename T>
    requires(N == 8 || N == 16 || N == 32 || N == 64) && std::is_arithmetic_v<T>
void sort_simd_network(std::span<T, N> data) noexcept {
    SIMDSortingNetwork<N, T>::sort(data);
}

} // namespace algoat::meta
