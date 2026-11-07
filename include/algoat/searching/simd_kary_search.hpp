/**
 * @file simd_kary_search.hpp
 * @brief Vectorized Multi-Way K-Ary Search Engine (AVX2 / AVX-512) and Static Multi-Way Search
 * Tree.
 *
 * Implements:
 * 1. simd_kary_search: Hybrid vector-assisted branchless lower_bound search over contiguous spans.
 * 2. SIMDKarySearchTree: Cache-line aligned static multi-way K-ary search tree for high-throughput
 * batch queries.
 * 3. SIMDKarySearch: SearchAlgorithm concept-compliant algorithm wrapper registered with Core
 * Registry.
 *
 * References:
 * - Benjamin Schlegel, Thomas Willhalm, Wolfgang Lehner (2011). "Fast sorted-set intersection using
 * SIMD instructions".
 * - Daniel Lemire, Leonid Boytsov, Nathan Kurz (2016). "SIMD compression and the intersection of
 * sorted integers".
 * - Modern C++ Dev (2023). "Vectorizing binary search with AVX2 and AVX-512".
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/searching/detail/prefetch.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(__cpp_modules)
// In C++20 module interface compilation, hardware vector builtins trigger compiler errors
// when imported in translation units without immintrin.h. Delegate to scalar fallback
// during module BMI compilation.
#define ALGOAT_SIMD_SEARCH_SCALAR_ONLY 1
#endif

#if !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY) && (defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace algoat::searching {

namespace detail {

/**
 * @struct AlignedAllocator
 * @brief Standard C++20 conforming allocator providing guaranteed byte alignment for heap buffers.
 * @tparam T Value type.
 * @tparam Alignment Byte alignment boundary (must be a power of two).
 */
template <typename T, std::size_t Alignment = 64> struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t alignment = Alignment;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    constexpr AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t{Alignment});
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept {
        if (p != nullptr) {
            ::operator delete(p, std::align_val_t{Alignment});
        }
    }

    template <typename U>
    constexpr bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    constexpr bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

/**
 * @brief Determines the sentinel value representing +infinity for type T.
 */
template <typename T> [[nodiscard]] constexpr T determine_sentinel() noexcept {
    if constexpr (std::numeric_limits<T>::is_specialized) {
        if constexpr (std::numeric_limits<T>::has_infinity) {
            return std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::max();
        }
    } else {
        return T{};
    }
}

/**
 * @brief Evaluates vector rank (number of elements where target > data[i]) using AVX-512.
 */
#if defined(__AVX512F__) && !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY)
template <typename T, std::size_t LANES, bool Aligned>
[[nodiscard]] inline std::size_t vector_rank_avx512(const T* ptr, const T& target) noexcept {
    if constexpr (LANES == 16) {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m512i vt = _mm512_set1_epi32(target);
            __m512i vd;
            if constexpr (Aligned) {
                vd = _mm512_load_si512(ptr);
            } else {
                vd = _mm512_loadu_si512(ptr);
            }
            __mmask16 mask = _mm512_cmpgt_epi32_mask(vt, vd);
            return static_cast<std::size_t>(std::popcount(static_cast<uint32_t>(mask)));
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            __m512i vt = _mm512_set1_epi32(static_cast<int32_t>(target));
            __m512i vd;
            if constexpr (Aligned) {
                vd = _mm512_load_si512(ptr);
            } else {
                vd = _mm512_loadu_si512(ptr);
            }
            __mmask16 mask = _mm512_cmpgt_epu32_mask(vt, vd);
            return static_cast<std::size_t>(std::popcount(static_cast<uint32_t>(mask)));
        } else if constexpr (std::is_same_v<T, float>) {
            __m512 vt = _mm512_set1_ps(target);
            __m512 vd;
            if constexpr (Aligned) {
                vd = _mm512_load_ps(ptr);
            } else {
                vd = _mm512_loadu_ps(ptr);
            }
            __mmask16 mask = _mm512_cmp_ps_mask(vt, vd, _CMP_GT_OQ);
            return static_cast<std::size_t>(std::popcount(static_cast<uint32_t>(mask)));
        }
    } else if constexpr (LANES == 8) {
        if constexpr (std::is_same_v<T, int64_t>) {
            __m512i vt = _mm512_set1_epi64(target);
            __m512i vd;
            if constexpr (Aligned) {
                vd = _mm512_load_si512(ptr);
            } else {
                vd = _mm512_loadu_si512(ptr);
            }
            __mmask8 mask = _mm512_cmpgt_epi64_mask(vt, vd);
            return static_cast<std::size_t>(std::popcount(static_cast<uint32_t>(mask)));
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            __m512i vt = _mm512_set1_epi64(static_cast<int64_t>(target));
            __m512i vd;
            if constexpr (Aligned) {
                vd = _mm512_load_si512(ptr);
            } else {
                vd = _mm512_loadu_si512(ptr);
            }
            __mmask8 mask = _mm512_cmpgt_epu64_mask(vt, vd);
            return static_cast<std::size_t>(std::popcount(static_cast<uint32_t>(mask)));
        } else if constexpr (std::is_same_v<T, double>) {
            __m512d vt = _mm512_set1_pd(target);
            __m512d vd;
            if constexpr (Aligned) {
                vd = _mm512_load_pd(ptr);
            } else {
                vd = _mm512_loadu_pd(ptr);
            }
            __mmask8 mask = _mm512_cmp_pd_mask(vt, vd, _CMP_GT_OQ);
            return static_cast<std::size_t>(std::popcount(static_cast<uint32_t>(mask)));
        }
    }
    std::size_t count = 0;
    for (std::size_t i = 0; i < LANES; ++i) {
        if (target > ptr[i])
            ++count;
    }
    return count;
}
#endif

/**
 * @brief Evaluates vector rank (number of elements where target > data[i]) using AVX2.
 */
#if defined(__AVX2__) && !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY)
template <typename T, std::size_t LANES, bool Aligned>
[[nodiscard]] inline std::size_t vector_rank_avx2(const T* ptr, const T& target) noexcept {
    if constexpr (LANES == 8) {
        if constexpr (std::is_same_v<T, int32_t>) {
            __m256i vt = _mm256_set1_epi32(target);
            __m256i vd;
            if constexpr (Aligned) {
                vd = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            } else {
                vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            }
            __m256i cmp = _mm256_cmpgt_epi32(vt, vd);
            uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) & 0xFF;
            return static_cast<std::size_t>(std::popcount(mask));
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            const __m256i bias = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
            __m256i vt = _mm256_xor_si256(_mm256_set1_epi32(static_cast<int32_t>(target)), bias);
            __m256i raw;
            if constexpr (Aligned) {
                raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            } else {
                raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            }
            __m256i vd = _mm256_xor_si256(raw, bias);
            __m256i cmp = _mm256_cmpgt_epi32(vt, vd);
            uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) & 0xFF;
            return static_cast<std::size_t>(std::popcount(mask));
        } else if constexpr (std::is_same_v<T, float>) {
            __m256 vt = _mm256_set1_ps(target);
            __m256 vd;
            if constexpr (Aligned) {
                vd = _mm256_load_ps(ptr);
            } else {
                vd = _mm256_loadu_ps(ptr);
            }
            __m256 cmp = _mm256_cmp_ps(vt, vd, _CMP_GT_OQ);
            uint32_t mask = _mm256_movemask_ps(cmp) & 0xFF;
            return static_cast<std::size_t>(std::popcount(mask));
        }
    } else if constexpr (LANES == 4) {
        if constexpr (std::is_same_v<T, int64_t>) {
            __m256i vt = _mm256_set1_epi64x(target);
            __m256i vd;
            if constexpr (Aligned) {
                vd = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            } else {
                vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            }
            __m256i cmp = _mm256_cmpgt_epi64(vt, vd);
            uint32_t mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) & 0x0F;
            return static_cast<std::size_t>(std::popcount(mask));
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            const __m256i bias = _mm256_set1_epi64x(static_cast<int64_t>(0x8000000000000000ULL));
            __m256i vt = _mm256_xor_si256(_mm256_set1_epi64x(static_cast<int64_t>(target)), bias);
            __m256i raw;
            if constexpr (Aligned) {
                raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            } else {
                raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            }
            __m256i vd = _mm256_xor_si256(raw, bias);
            __m256i cmp = _mm256_cmpgt_epi64(vt, vd);
            uint32_t mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) & 0x0F;
            return static_cast<std::size_t>(std::popcount(mask));
        } else if constexpr (std::is_same_v<T, double>) {
            __m256d vt = _mm256_set1_pd(target);
            __m256d vd;
            if constexpr (Aligned) {
                vd = _mm256_load_pd(ptr);
            } else {
                vd = _mm256_loadu_pd(ptr);
            }
            __m256d cmp = _mm256_cmp_pd(vt, vd, _CMP_GT_OQ);
            uint32_t mask = _mm256_movemask_pd(cmp) & 0x0F;
            return static_cast<std::size_t>(std::popcount(mask));
        }
    }
    std::size_t count = 0;
    for (std::size_t i = 0; i < LANES; ++i) {
        if (target > ptr[i])
            ++count;
    }
    return count;
}
#endif

/**
 * @brief Generic dispatch for vector rank counting.
 */
template <typename T, std::size_t LANES, bool Aligned = false>
[[nodiscard]] inline std::size_t evaluate_vector_rank(const T* ptr, const T& target) noexcept {
#if defined(__AVX512F__) && !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY)
    if constexpr ((LANES == 16 && (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                   std::is_same_v<T, float>)) ||
                  (LANES == 8 && (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                                  std::is_same_v<T, double>))) {
        return vector_rank_avx512<T, LANES, Aligned>(ptr, target);
    }
#endif
#if defined(__AVX2__) && !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY)
    if constexpr ((LANES == 8 && (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                  std::is_same_v<T, float>)) ||
                  (LANES == 4 && (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
                                  std::is_same_v<T, double>))) {
        return vector_rank_avx2<T, LANES, Aligned>(ptr, target);
    }
#endif
    std::size_t count = 0;
    for (std::size_t i = 0; i < LANES; ++i) {
        if (target > ptr[i])
            ++count;
    }
    return count;
}

/**
 * @brief Determines the optimal SIMD vector lane count for type T.
 */
template <typename T> consteval std::size_t optimal_vector_lanes() noexcept {
#if defined(__AVX512F__) && !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY)
    if constexpr (sizeof(T) == 4)
        return 16;
    if constexpr (sizeof(T) == 8)
        return 8;
#elif defined(__AVX2__) && !defined(ALGOAT_SIMD_SEARCH_SCALAR_ONLY)
    if constexpr (sizeof(T) == 4)
        return 8;
    if constexpr (sizeof(T) == 8)
        return 4;
#endif
    return 1;
}

} // namespace detail

/**
 * @brief SIMD multi-way / vector-accelerated lower_bound search over a flat contiguous span.
 *
 * Implements a hybrid branchless vector architecture: uses branchless binary reduction
 * down to small cache-resident windows, then performs unrolled contiguous SIMD evaluation.
 *
 * @tparam T Arithmetic element type.
 * @param data Contiguous sorted span of elements.
 * @param target Value to search for.
 * @return 0-based index of the first element >= target, or data.size() if all elements are <
 * target.
 */
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] std::size_t simd_kary_search(std::span<const T> data, const T& target) noexcept {
    const std::size_t n = data.size();
    if (n == 0)
        return 0;

    const T* base = data.data();
    std::size_t len = n;

    constexpr std::size_t LANES = detail::optimal_vector_lanes<T>();

    if constexpr (LANES > 1) {
        constexpr std::size_t THRESHOLD = LANES * 2;

        while (len > THRESHOLD) {
            std::size_t half = len >> 1;
            const T* mid = base + half - 1;
            base = (*mid < target) ? (base + half) : base;
            len -= half;
        }

        std::size_t i = 0;
        for (; i + LANES <= len; i += LANES) {
            std::size_t pop = detail::evaluate_vector_rank<T, LANES, false>(base + i, target);
            if (pop < LANES) {
                return static_cast<std::size_t>((base - data.data()) + i + pop);
            }
        }
        for (; i < len; ++i) {
            if (base[i] >= target) {
                return static_cast<std::size_t>((base - data.data()) + i);
            }
        }
        return static_cast<std::size_t>((base - data.data()) + len);
    } else {
        while (len > 0) {
            std::size_t half = len >> 1;
            const T* mid = base + half;
            if (*mid < target) {
                base = mid + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        return static_cast<std::size_t>(base - data.data());
    }
}

/**
 * @class SIMDKarySearchTree
 * @brief Cache-line aligned static multi-way K-ary search tree for high-throughput batch queries.
 *
 * Packs K - 1 splitters + 1 sentinel per node into 64-byte aligned memory blocks matching SIMD
 * registers and CPU cache lines. Reduces tree height to log_K(N), evaluating multi-way child
 * routing in single clock cycles via vector comparisons and hardware popcounts.
 *
 * @tparam T Arithmetic element type.
 * @tparam K Fanout branching factor (must be a power of 2 >= 2, default 8).
 */
template <typename T, std::size_t K = 8> class SIMDKarySearchTree {
    static_assert(K >= 2 && (K & (K - 1)) == 0, "Fanout K must be a power of 2 >= 2");
    static_assert(std::is_arithmetic_v<T>, "Element type T must be an arithmetic type");

public:
    using value_type = T;
    static constexpr std::size_t FANOUT = K;
    static constexpr std::size_t KEYS = K - 1;

    /**
     * @brief Default constructor creating an empty search tree.
     */
    SIMDKarySearchTree() = default;

    /**
     * @brief Constructs a SIMDKarySearchTree from any contiguous range in O(N) time.
     * @tparam R Range type satisfying std::ranges::contiguous_range.
     * @param data Range of elements sorted in non-decreasing order.
     */
    template <std::ranges::contiguous_range R> explicit SIMDKarySearchTree(R&& data) {
        build(std::span<const T>{std::ranges::data(data), std::ranges::size(data)});
    }

    /**
     * @brief Constructs the static K-ary search tree from a sorted span of data in O(N) time.
     * @param data Range of elements sorted in non-decreasing order.
     */
    void build(std::span<const T> data) {
        n_ = data.size();
        sentinel_ = detail::determine_sentinel<T>();

        if (n_ == 0) {
            tree_.clear();
            layers_.clear();
            return;
        }

        // 1. Calculate leaf layer capacity
        const std::size_t num_leaves = (n_ + KEYS - 1) / KEYS;

        std::vector<std::size_t> layer_counts;
        layer_counts.push_back(num_leaves);
        while (layer_counts.back() > 1) {
            std::size_t next_count = (layer_counts.back() + FANOUT - 1) / FANOUT;
            layer_counts.push_back(next_count);
        }

        std::reverse(layer_counts.begin(), layer_counts.end());
        const std::size_t height = layer_counts.size();

        layers_.resize(height);
        std::size_t total_nodes = 0;
        for (std::size_t l = 0; l < height; ++l) {
            layers_[l].offset = total_nodes;
            layers_[l].count = layer_counts[l];
            total_nodes += layer_counts[l];
        }

        // Allocate nodes (+1 safety sentinel node at end) with guaranteed 64-byte alignment
        tree_.assign((total_nodes + 1) * FANOUT, sentinel_);

        // 2. Populate leaf level
        const std::size_t leaf_offset = layers_.back().offset;
        for (std::size_t b = 0; b < num_leaves; ++b) {
            T* leaf_node = &tree_[(leaf_offset + b) * FANOUT];
            for (std::size_t j = 0; j < KEYS; ++j) {
                std::size_t src_idx = b * KEYS + j;
                if (src_idx < n_) {
                    leaf_node[j] = data[src_idx];
                } else {
                    leaf_node[j] = sentinel_;
                }
            }
            leaf_node[KEYS] = sentinel_;
        }

        // Subtree maximum keys for the layer directly beneath the current one being built
        std::vector<T> child_max_keys(num_leaves);
        for (std::size_t b = 0; b < num_leaves; ++b) {
            std::size_t last_elem_idx = std::min(n_ - 1, (b + 1) * KEYS - 1);
            child_max_keys[b] = data[last_elem_idx];
        }

        // 3. Build internal levels bottom-up from level height - 2 down to 0 (root)
        for (std::size_t l = height - 1; l > 0; --l) {
            const std::size_t parent_level = l - 1;
            const std::size_t parent_offset = layers_[parent_level].offset;
            const std::size_t num_parents = layers_[parent_level].count;
            const std::size_t child_count = child_max_keys.size();

            std::vector<T> parent_max_keys(num_parents);

            for (std::size_t p = 0; p < num_parents; ++p) {
                T* parent_node = &tree_[(parent_offset + p) * FANOUT];
                for (std::size_t j = 0; j < KEYS; ++j) {
                    std::size_t child_idx = p * FANOUT + j;
                    if (child_idx < child_count) {
                        parent_node[j] = child_max_keys[child_idx];
                    } else {
                        parent_node[j] = sentinel_;
                    }
                }
                parent_node[KEYS] = sentinel_;

                std::size_t last_child_idx = std::min(child_count - 1, (p + 1) * FANOUT - 1);
                parent_max_keys[p] = child_max_keys[last_child_idx];
            }

            child_max_keys = std::move(parent_max_keys);
        }
    }

    /**
     * @brief Performs a SIMD K-ary lower_bound search for a single target with pipelined
     * prefetching.
     * @param target Search key value.
     * @return 0-based index of the first element >= target, or size() if none found.
     */
    [[nodiscard]] std::size_t lower_bound(const T& target) const noexcept {
        if (n_ == 0)
            return 0;

        std::size_t block_idx = 0;
        const std::size_t height = layers_.size();
        constexpr std::size_t fanout_shift = std::countr_zero(FANOUT);

        // Traverse internal levels from root (level 0) down to parents of leaves
        for (std::size_t level = 0; level + 1 < height; ++level) {
            if (block_idx >= layers_[level].count) {
                return n_;
            }
            const T* node_ptr = &tree_[(layers_[level].offset + block_idx) * FANOUT];

            std::size_t child_rank = evaluate_node_rank(node_ptr, target);
            block_idx = (block_idx << fanout_shift) + child_rank;

            // Early child address prefetching pipelined while loop counter increments
            const std::size_t next_level = level + 1;
            if (next_level < height && block_idx < layers_[next_level].count) {
                const T* next_node_ptr = &tree_[(layers_[next_level].offset + block_idx) * FANOUT];
                detail::prefetch_read(next_node_ptr);
            }
        }

        // Final leaf level
        if (block_idx >= layers_.back().count) {
            return n_;
        }
        const T* leaf_ptr = &tree_[(layers_.back().offset + block_idx) * FANOUT];
        std::size_t leaf_rank = evaluate_node_rank(leaf_ptr, target);

        std::size_t sorted_idx = (block_idx * KEYS) + leaf_rank;
        return (sorted_idx < n_) ? sorted_idx : n_;
    }

    /**
     * @brief Performs an exact key lookup in the tree.
     * @param target Search key value.
     * @return 0-based index if target is found, or std::nullopt.
     */
    [[nodiscard]] std::optional<std::size_t> find(const T& target) const noexcept {
        if (n_ == 0)
            return std::nullopt;
        std::size_t idx = lower_bound(target);
        if (idx < n_) {
            const std::size_t leaf_node_idx = layers_.back().offset + (idx / KEYS);
            const std::size_t slot_idx = idx % KEYS;
            if (tree_[leaf_node_idx * FANOUT + slot_idx] == target) {
                return idx;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Returns the total number of elements in the tree.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return n_;
    }

private:
    struct LayerInfo {
        std::size_t offset{0};
        std::size_t count{0};
    };

    // Cache-line aligned storage: nodes reside at 64-byte boundaries on heap
    alignas(64) std::vector<T, detail::AlignedAllocator<T, 64>> tree_;
    std::size_t n_{0};
    std::vector<LayerInfo> layers_;
    T sentinel_{};

    [[nodiscard]] inline std::size_t evaluate_node_rank(const T* node_ptr,
                                                        const T& target) const noexcept {
        // Nodes are packed with KEYS splitters + 1 sentinel at index KEYS.
        // Evaluating vector_rank over FANOUT elements counts splitters < target,
        // since sentinel >= target always yields false for slot KEYS.
        std::size_t rank = detail::evaluate_vector_rank<T, FANOUT, true>(node_ptr, target);
        return std::min(rank, KEYS);
    }
};

/**
 * @struct SIMDKarySearch
 * @brief SearchAlgorithm concept-compliant wrapper for SIMD K-ary search.
 */
struct SIMDKarySearch {
    /**
     * @brief Returns the unique lowercase string identifier for the algorithm.
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "simdkarysearch";
    }

    /**
     * @brief Performs SIMD K-ary search over a mutable contiguous span.
     */
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] std::optional<std::size_t> search(std::span<T> data,
                                                    const T& target) const noexcept {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }

    /**
     * @brief Performs SIMD K-ary search over a const contiguous span.
     */
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] std::optional<std::size_t> search(std::span<const T> data,
                                                    const T& target) const noexcept {
        if (data.empty())
            return std::nullopt;
        std::size_t idx = simd_kary_search(data, target);
        if (idx < data.size() && data[idx] == target) {
            return idx;
        }
        return std::nullopt;
    }

    /**
     * @brief Indicates whether the algorithm requires its input span to be sorted.
     */
    [[nodiscard]] static constexpr bool requires_sorted() noexcept {
        return true;
    }
};

} // namespace algoat::searching

ALGOAT_REGISTER_ALGORITHM("searching", "simdkarysearch", ::algoat::searching::SIMDKarySearch)
