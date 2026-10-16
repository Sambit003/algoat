/**
 * @file static_btree.hpp
 * @brief Cache-line aligned Static B-Tree (S+ Tree layout) with SIMD vector in-node search.
 *
 * References:
 * - Sergey Slotin (2022), "Algorithms for Modern Hardware", Static B-Trees.
 * - Ragnar Groot Koerkamp (2024), "Static search trees: 40x faster than binary search".
 * - Rao & Ross (1999), "Cache-Conscious Index Structures for Main-Memory Databases" (CSS-Trees).
 * - Kim et al. (2010), "FAST: Fast Architecture Sensitive Tree Search on Modern CPUs and GPUs"
 * (SIGMOD).
 */

#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "algoat/searching/detail/prefetch.hpp"

namespace algoat::searching {

namespace detail {

template <typename T> struct alignas(64) StaticBTreeNode {
    static constexpr std::size_t CAPACITY = 16;
    static constexpr std::size_t KEYS = 15;
    T data[CAPACITY];

    [[nodiscard]] unsigned rank(const T& target) const noexcept;
};

#if defined(__AVX2__)
[[nodiscard]] inline unsigned reduce_avx2_mask(int ma, int mb) noexcept {
    unsigned mask = static_cast<unsigned>(ma) | (static_cast<unsigned>(mb) << 8);
    return static_cast<unsigned>(std::popcount(mask));
}

[[nodiscard]] inline unsigned rank_avx2_i32(const __m256i q_splat, const int32_t* node) noexcept {
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(node));
    __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(node + 8));

    __m256i ca = _mm256_cmpgt_epi32(q_splat, a);
    __m256i cb = _mm256_cmpgt_epi32(q_splat, b);

    return reduce_avx2_mask(_mm256_movemask_ps(_mm256_castsi256_ps(ca)),
                            _mm256_movemask_ps(_mm256_castsi256_ps(cb)));
}

[[nodiscard]] inline unsigned rank_avx2_u32(const __m256i q_splat_bias,
                                            const uint32_t* node) noexcept {
    const __m256i bias = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(node));
    __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(node + 8));

    a = _mm256_xor_si256(a, bias);
    b = _mm256_xor_si256(b, bias);

    __m256i ca = _mm256_cmpgt_epi32(q_splat_bias, a);
    __m256i cb = _mm256_cmpgt_epi32(q_splat_bias, b);

    return reduce_avx2_mask(_mm256_movemask_ps(_mm256_castsi256_ps(ca)),
                            _mm256_movemask_ps(_mm256_castsi256_ps(cb)));
}
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
[[nodiscard]] inline unsigned reduce_neon_sum(int32x4_t total) noexcept {
#if defined(__aarch64__)
    return static_cast<unsigned>(-vaddvq_s32(total));
#else
    int32x2_t r = vpadd_s32(vget_low_s32(total), vget_high_s32(total));
    r = vpadd_s32(r, r);
    return static_cast<unsigned>(-vget_lane_s32(r, 0));
#endif
}

[[nodiscard]] inline unsigned rank_neon_i32(int32x4_t q_splat, const int32_t* node) noexcept {
    int32x4_t v0 = vld1q_s32(node);
    int32x4_t v1 = vld1q_s32(node + 4);
    int32x4_t v2 = vld1q_s32(node + 8);
    int32x4_t v3 = vld1q_s32(node + 12);

    int32x4_t c0 = vreinterpretq_s32_u32(vcgtq_s32(q_splat, v0));
    int32x4_t c1 = vreinterpretq_s32_u32(vcgtq_s32(q_splat, v1));
    int32x4_t c2 = vreinterpretq_s32_u32(vcgtq_s32(q_splat, v2));
    int32x4_t c3 = vreinterpretq_s32_u32(vcgtq_s32(q_splat, v3));

    int32x4_t s01 = vaddq_s32(c0, c1);
    int32x4_t s23 = vaddq_s32(c2, c3);
    return reduce_neon_sum(vaddq_s32(s01, s23));
}

[[nodiscard]] inline unsigned rank_neon_u32(uint32x4_t q_splat, const uint32_t* node) noexcept {
    uint32x4_t v0 = vld1q_u32(node);
    uint32x4_t v1 = vld1q_u32(node + 4);
    uint32x4_t v2 = vld1q_u32(node + 8);
    uint32x4_t v3 = vld1q_u32(node + 12);

    int32x4_t c0 = vreinterpretq_s32_u32(vcgtq_u32(q_splat, v0));
    int32x4_t c1 = vreinterpretq_s32_u32(vcgtq_u32(q_splat, v1));
    int32x4_t c2 = vreinterpretq_s32_u32(vcgtq_u32(q_splat, v2));
    int32x4_t c3 = vreinterpretq_s32_u32(vcgtq_u32(q_splat, v3));

    int32x4_t s01 = vaddq_s32(c0, c1);
    int32x4_t s23 = vaddq_s32(c2, c3);
    return reduce_neon_sum(vaddq_s32(s01, s23));
}
#endif

template <typename T>
[[nodiscard]] inline unsigned StaticBTreeNode<T>::rank(const T& target) const noexcept {
#if defined(__AVX2__)
    if constexpr (std::same_as<T, int32_t>) {
        __m256i q = _mm256_set1_epi32(target);
        return rank_avx2_i32(q, reinterpret_cast<const int32_t*>(data));
    } else if constexpr (std::same_as<T, uint32_t>) {
        __m256i bias = _mm256_set1_epi32(static_cast<int32_t>(0x80000000u));
        __m256i q = _mm256_xor_si256(_mm256_set1_epi32(static_cast<int32_t>(target)), bias);
        return rank_avx2_u32(q, reinterpret_cast<const uint32_t*>(data));
    } else
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
        if constexpr (std::same_as<T, int32_t>) {
        int32x4_t q = vdupq_n_s32(target);
        return rank_neon_i32(q, reinterpret_cast<const int32_t*>(data));
    } else if constexpr (std::same_as<T, uint32_t>) {
        uint32x4_t q = vdupq_n_u32(target);
        return rank_neon_u32(q, reinterpret_cast<const uint32_t*>(data));
    } else
#endif
    {
        unsigned count = 0;
        for (std::size_t i = 0; i < KEYS; ++i) {
            if (data[i] < target) {
                ++count;
            }
        }
        return count;
    }
}

} // namespace detail

/**
 * @class StaticBTree
 * @brief Cache-line aligned Static B-Tree (S+ Tree layout) with SIMD vector in-node search.
 *
 * Implements a static B-tree with branching factor B = 15 keys + 1 sentinel in a 64-byte
 * cache-line node. Internal levels route searches via "left-max" keys, eliminating boundary
 * cache-line fetches. Supports both single-query search and batched query search interleaving
 * memory prefetch requests to saturate CPU Line Fill Buffers.
 *
 * @tparam T Element type satisfying std::totally_ordered.
 */
template <std::totally_ordered T> class StaticBTree {
public:
    using Node = detail::StaticBTreeNode<T>;
    static constexpr std::size_t B = Node::KEYS;          // 15
    static constexpr std::size_t FANOUT = Node::CAPACITY; // 16

    /**
     * @brief Constructs an S+ Static B-Tree from a sorted span of data.
     * @param sorted_data Range of elements sorted in non-decreasing order.
     */
    explicit StaticBTree(std::span<const T> sorted_data)
        : size_(sorted_data.size()), sentinel_(determine_sentinel(sorted_data)) {
        if (size_ == 0) {
            return;
        }

        build_tree(sorted_data);
    }

    /**
     * @brief Performs a SIMD-accelerated lower bound search for a single key.
     * @param target Search key value.
     * @return 0-based index of the first element >= target in the sorted data,
     *         or size() if all elements are < target.
     */
    [[nodiscard]] std::size_t lower_bound(const T& target) const noexcept {
        if (size_ == 0) {
            return 0;
        }

        std::size_t block_idx = 0;
        const std::size_t height = layers_.size();

        // Traverse internal levels from root (level 0) down to parents of leaves
        for (std::size_t level = 0; level + 1 < height; ++level) {
            if (block_idx >= layers_[level].count) {
                return size_;
            }
            const Node& node = nodes_[layers_[level].offset + block_idx];
            unsigned child_rank = node.rank(target);
            block_idx = (block_idx << 4) + child_rank;
        }

        // Final leaf level
        if (block_idx >= layers_.back().count) {
            return size_;
        }
        const Node& leaf_node = nodes_[layers_.back().offset + block_idx];
        unsigned leaf_rank = leaf_node.rank(target);

        return resolve_leaf_index(block_idx, leaf_rank);
    }

    /**
     * @brief Batched query interface interleaving lookups to saturate CPU Line Fill Buffers.
     * @param targets Span of query keys.
     * @param results Span receiving the resulting 0-based lower bound indices.
     */
    void lower_bound_batch(std::span<const T> targets,
                           std::span<std::size_t> results) const noexcept {
        const std::size_t num_queries = targets.size();
        if (num_queries == 0) {
            return;
        }

        if (size_ == 0) {
            for (std::size_t i = 0; i < num_queries; ++i) {
                results[i] = 0;
            }
            return;
        }

        constexpr std::size_t INTERLEAVED_WAYS = 8; // Batch size matching CPU Line Fill Buffers
        const std::size_t height = layers_.size();
        const std::size_t sentinel_node_idx = nodes_.size() - 1;
        std::size_t i = 0;

        for (; i + INTERLEAVED_WAYS <= num_queries; i += INTERLEAVED_WAYS) {
            std::size_t child_blocks[INTERLEAVED_WAYS] = {0};

            // Traverse internal levels in lockstep across the batch
            for (std::size_t level = 0; level + 1 < height; ++level) {
                const std::size_t current_layer_offset = layers_[level].offset;
                const std::size_t next_layer_offset = layers_[level + 1].offset;
                const std::size_t current_limit = layers_[level].count;
                const std::size_t next_limit = layers_[level + 1].count;

                for (std::size_t w = 0; w < INTERLEAVED_WAYS; ++w) {
                    const std::size_t node_idx = (child_blocks[w] < current_limit)
                                                     ? (current_layer_offset + child_blocks[w])
                                                     : sentinel_node_idx;
                    const Node& current_node = nodes_[node_idx];
                    unsigned child_rank = current_node.rank(targets[i + w]);
                    child_blocks[w] = (child_blocks[w] << 4) + child_rank;

                    // Prefetch child node for next level
                    const std::size_t next_node_idx = (child_blocks[w] < next_limit)
                                                          ? (next_layer_offset + child_blocks[w])
                                                          : sentinel_node_idx;
                    detail::prefetch_read(&nodes_[next_node_idx]);
                }
            }

            // Resolve leaf level
            const std::size_t leaf_layer_offset = layers_.back().offset;
            const std::size_t leaf_limit = layers_.back().count;
            for (std::size_t w = 0; w < INTERLEAVED_WAYS; ++w) {
                if (child_blocks[w] >= leaf_limit) {
                    results[i + w] = size_;
                } else {
                    const Node& leaf_node = nodes_[leaf_layer_offset + child_blocks[w]];
                    unsigned leaf_rank = leaf_node.rank(targets[i + w]);
                    results[i + w] = resolve_leaf_index(child_blocks[w], leaf_rank);
                }
            }
        }

        // Process leftovers sequentially
        for (; i < num_queries; ++i) {
            results[i] = lower_bound(targets[i]);
        }
    }

    /**
     * @brief Returns the total number of elements in the tree.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

private:
    struct LayerInfo {
        std::size_t offset{0};
        std::size_t count{0};
    };

    std::size_t size_{0};
    T sentinel_{};
    std::vector<Node> nodes_;
    std::vector<LayerInfo> layers_;

    [[nodiscard]] inline std::size_t resolve_leaf_index(std::size_t leaf_block,
                                                        unsigned leaf_rank) const noexcept {
        const std::size_t sorted_idx = (leaf_block * B) + leaf_rank;
        return (sorted_idx < size_) ? sorted_idx : size_;
    }

    static T determine_sentinel(std::span<const T> sorted_data) {
        if constexpr (std::numeric_limits<T>::is_specialized) {
            if constexpr (std::numeric_limits<T>::has_infinity) {
                return std::numeric_limits<T>::infinity();
            } else {
                return std::numeric_limits<T>::max();
            }
        } else {
            return sorted_data.empty() ? T{} : sorted_data.back();
        }
    }

    void build_tree(std::span<const T> sorted_data) {
        // 1. Calculate number of leaf blocks
        const std::size_t num_leaves = (size_ + B - 1) / B;

        // Calculate layer block counts bottom-up
        std::vector<std::size_t> layer_counts;
        layer_counts.push_back(num_leaves);
        while (layer_counts.back() > 1) {
            std::size_t next_count = (layer_counts.back() + FANOUT - 1) / FANOUT;
            layer_counts.push_back(next_count);
        }

        // Reverse layer counts so layer 0 is Root, layer H - 1 is Leaves
        std::reverse(layer_counts.begin(), layer_counts.end());
        const std::size_t height = layer_counts.size();

        // Compute layer offsets and allocate nodes (+1 sentinel safety node at end)
        layers_.resize(height);
        std::size_t total_nodes = 0;
        for (std::size_t l = 0; l < height; ++l) {
            layers_[l].offset = total_nodes;
            layers_[l].count = layer_counts[l];
            total_nodes += layer_counts[l];
        }

        nodes_.resize(total_nodes + 1); // +1 safety sentinel node
        for (std::size_t j = 0; j < Node::CAPACITY; ++j) {
            nodes_.back().data[j] = sentinel_;
        }

        // 2. Populate Leaf level (level height - 1)
        const std::size_t leaf_offset = layers_.back().offset;
        for (std::size_t b = 0; b < num_leaves; ++b) {
            Node& leaf = nodes_[leaf_offset + b];
            for (std::size_t j = 0; j < B; ++j) {
                std::size_t src_idx = b * B + j;
                if (src_idx < size_) {
                    leaf.data[j] = sorted_data[src_idx];
                } else {
                    leaf.data[j] = sentinel_;
                }
            }
            leaf.data[15] = sentinel_;
        }

        // Subtree maximum keys for the layer directly beneath the current one being built
        std::vector<T> child_max_keys(num_leaves);
        for (std::size_t b = 0; b < num_leaves; ++b) {
            std::size_t last_elem_idx = std::min(size_ - 1, (b + 1) * B - 1);
            child_max_keys[b] = sorted_data[last_elem_idx];
        }

        // 3. Build internal levels bottom-up from level height - 2 down to 0 (root)
        for (int level = static_cast<int>(height) - 2; level >= 0; --level) {
            const std::size_t current_count = layers_[level].count;
            const std::size_t num_children = child_max_keys.size();
            const std::size_t current_offset = layers_[level].offset;

            std::vector<T> current_max_keys(current_count, sentinel_);

            for (std::size_t i = 0; i < current_count; ++i) {
                Node& node = nodes_[current_offset + i];
                for (std::size_t j = 0; j < B; ++j) {
                    std::size_t child_idx = (i << 4) + j;
                    if (child_idx < num_children) {
                        node.data[j] = child_max_keys[child_idx];
                    } else {
                        node.data[j] = sentinel_;
                    }
                }
                node.data[15] = sentinel_;

                // Max key of subtree rooted at this node is the max key of its last valid child
                std::size_t last_child = std::min(num_children - 1, ((i + 1) << 4) - 1);
                if (i * FANOUT < num_children) {
                    current_max_keys[i] = child_max_keys[last_child];
                } else {
                    current_max_keys[i] = sentinel_;
                }
            }

            child_max_keys = std::move(current_max_keys);
        }
    }
};

} // namespace algoat::searching
