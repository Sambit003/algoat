/**
 * @file d_ary_heap.hpp
 * @brief Cache-aligned d-ary indexed min-heap with AVX2 SIMD min-reduction.
 *
 * Implements an in-place indexed priority queue with O(1) position lookup, O(log_d N)
 * sift-up / decrease-key, O(d log_d N) sift-down / pop, 64-byte aligned contiguous
 * memory layout, and AVX2 SIMD min-reductions across 4-child groups.
 */

#pragma once

#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace algoat::graph {

namespace detail {

#if defined(__AVX2__)
inline std::pair<double, int> argmin4_avx2_double(const double* ptr) noexcept {
    __m256d v = _mm256_loadu_pd(ptr);
    __m256d s1 = _mm256_permute4x64_pd(v, _MM_SHUFFLE(1, 0, 3, 2));
    __m256d m1 = _mm256_min_pd(v, s1);
    __m256d s2 = _mm256_shuffle_pd(m1, m1, 0b0101);
    __m256d min_all = _mm256_min_pd(m1, s2);
    __m256d cmp = _mm256_cmp_pd(v, min_all, _CMP_EQ_OQ);
    int mask = _mm256_movemask_pd(cmp);
    int min_idx = std::countr_zero(static_cast<unsigned int>(mask));
    double min_val = _mm256_cvtsd_f64(min_all);
    return {min_val, min_idx};
}

inline std::pair<float, int> argmin4_avx2_float(const float* ptr) noexcept {
    __m128 v = _mm_loadu_ps(ptr);
    __m128 s1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 m1 = _mm_min_ps(v, s1);
    __m128 s2 = _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 min_all = _mm_min_ps(m1, s2);
    __m128 cmp = _mm_cmpeq_ps(v, min_all);
    int mask = _mm_movemask_ps(cmp);
    int min_idx = std::countr_zero(static_cast<unsigned int>(mask));
    float min_val = _mm_cvtss_f32(min_all);
    return {min_val, min_idx};
}
#endif

template <typename T>
inline std::pair<T, int> argmin4_scalar_branchless(const T* p, std::size_t count) noexcept {
    int best_idx = 0;
    T best_val = p[0];
    for (std::size_t i = 1; i < count; ++i) {
        if (p[i] < best_val) {
            best_val = p[i];
            best_idx = static_cast<int>(i);
        }
    }
    return {best_val, best_idx};
}

} // namespace detail

/**
 * @class d_ary_heap
 * @brief High-performance d-ary min-heap with indexed in-place priority updates.
 *
 * @tparam Arity Branching factor (default 4, must be >= 2).
 * @tparam KeyType Priority type (e.g. double, float, int64_t).
 * @tparam IndexType Identifier type (e.g. VertexId / uint32_t).
 */
template <std::size_t Arity = 4, typename KeyType = double, typename IndexType = VertexId>
class d_ary_heap {
    static_assert(Arity >= 2, "d_ary_heap arity must be at least 2");

public:
    using key_type = KeyType;
    using index_type = IndexType;

    static constexpr int32_t NOT_IN_HEAP = -1;
    static constexpr int32_t SETTLED = -2;

    explicit d_ary_heap(std::size_t initial_capacity = 0) {
        if (initial_capacity > 0) {
            reserve(initial_capacity);
        }
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return priorities_.size();
    }

    [[nodiscard]] bool contains(IndexType id) const noexcept {
        const auto id_sz = static_cast<std::size_t>(id);
        return id_sz < pos_in_heap_.size() && pos_in_heap_[id_sz] >= 0;
    }

    [[nodiscard]] bool is_settled(IndexType id) const noexcept {
        const auto id_sz = static_cast<std::size_t>(id);
        return id_sz < pos_in_heap_.size() && pos_in_heap_[id_sz] == SETTLED;
    }

    [[nodiscard]] KeyType get_key(IndexType id) const {
        if (!contains(id)) {
            throw std::out_of_range("Identifier is not currently in the heap");
        }
        return priorities_[static_cast<std::size_t>(pos_in_heap_[static_cast<std::size_t>(id)])];
    }

    [[nodiscard]] IndexType top_index() const {
        if (empty()) {
            throw std::underflow_error("top_index() called on empty d_ary_heap");
        }
        return vertices_[0];
    }

    [[nodiscard]] KeyType top_key() const {
        if (empty()) {
            throw std::underflow_error("top_key() called on empty d_ary_heap");
        }
        return priorities_[0];
    }

    void reserve(std::size_t capacity) {
        priorities_.reserve(capacity);
        vertices_.reserve(capacity);
        if (capacity > pos_in_heap_.size()) {
            pos_in_heap_.resize(capacity, NOT_IN_HEAP);
        }
    }

    void reset(std::size_t num_elements) {
        clear();
        pos_in_heap_.assign(num_elements, NOT_IN_HEAP);
    }

    void clear() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            const auto v = static_cast<std::size_t>(vertices_[i]);
            if (v < pos_in_heap_.size()) {
                pos_in_heap_[v] = NOT_IN_HEAP;
            }
        }
        size_ = 0;
        priorities_.clear();
        vertices_.clear();
    }

    void push(IndexType id, KeyType priority) {
        const auto id_sz = static_cast<std::size_t>(id);
        if (id_sz >= pos_in_heap_.size()) {
            pos_in_heap_.resize(id_sz + 1, NOT_IN_HEAP);
        }

        const std::size_t hole = size_;
        ++size_;
        if (hole >= priorities_.size()) {
            priorities_.push_back(priority);
            vertices_.push_back(id);
        } else {
            priorities_[hole] = priority;
            vertices_[hole] = id;
        }
        pos_in_heap_[id_sz] = static_cast<int32_t>(hole);
        sift_up(hole);
    }

    void decrease_key(IndexType id, KeyType new_priority) {
        const auto id_sz = static_cast<std::size_t>(id);
        if (id_sz >= pos_in_heap_.size() || pos_in_heap_[id_sz] < 0) {
            throw std::invalid_argument("Cannot decrease key for element not in heap");
        }

        const auto idx = static_cast<std::size_t>(pos_in_heap_[id_sz]);
        if (new_priority > priorities_[idx]) {
            return; // Not a decrease
        }

        priorities_[idx] = new_priority;
        sift_up(idx);
    }

    void push_or_decrease_key(IndexType id, KeyType priority) {
        const auto id_sz = static_cast<std::size_t>(id);
        if (id_sz >= pos_in_heap_.size() || pos_in_heap_[id_sz] == NOT_IN_HEAP) {
            push(id, priority);
        } else if (pos_in_heap_[id_sz] >= 0) {
            const auto idx = static_cast<std::size_t>(pos_in_heap_[id_sz]);
            if (priority < priorities_[idx]) {
                priorities_[idx] = priority;
                sift_up(idx);
            }
        }
    }

    std::pair<IndexType, KeyType> pop() {
        if (empty()) {
            throw std::underflow_error("pop() called on empty d_ary_heap");
        }

        const IndexType min_id = vertices_[0];
        const KeyType min_key = priorities_[0];
        const auto min_id_sz = static_cast<std::size_t>(min_id);
        if (min_id_sz < pos_in_heap_.size()) {
            pos_in_heap_[min_id_sz] = SETTLED;
        }

        --size_;
        if (size_ > 0) {
            priorities_[0] = priorities_[size_];
            vertices_[0] = vertices_[size_];
            pos_in_heap_[static_cast<std::size_t>(vertices_[0])] = 0;
            sift_down(0);
        }

        return {min_id, min_key};
    }

private:
    static constexpr std::size_t parent(std::size_t i) noexcept {
        if constexpr (Arity == 4) {
            return (i - 1) >> 2;
        } else {
            return (i - 1) / Arity;
        }
    }

    static constexpr std::size_t first_child(std::size_t i) noexcept {
        if constexpr (Arity == 4) {
            return (i << 2) + 1;
        } else {
            return Arity * i + 1;
        }
    }

    void sift_up(std::size_t hole) {
        const KeyType val = priorities_[hole];
        const IndexType id = vertices_[hole];

        while (hole > 0) {
            const std::size_t p = parent(hole);
            if (val >= priorities_[p]) {
                break;
            }

            priorities_[hole] = priorities_[p];
            vertices_[hole] = vertices_[p];
            pos_in_heap_[static_cast<std::size_t>(vertices_[hole])] = static_cast<int32_t>(hole);
            hole = p;
        }

        priorities_[hole] = val;
        vertices_[hole] = id;
        pos_in_heap_[static_cast<std::size_t>(id)] = static_cast<int32_t>(hole);
    }

    void sift_down(std::size_t hole) {
        const KeyType val = priorities_[hole];
        const IndexType id = vertices_[hole];

        while (true) {
            const std::size_t child_start = first_child(hole);
            if (child_start >= size_) {
                break;
            }

            KeyType best_child_val;
            int best_child_offset = 0;

            if constexpr (Arity == 4) {
#if defined(__AVX2__)
                if constexpr (std::is_same_v<KeyType, double>) {
                    if (child_start + 4 <= size_) {
                        auto [m_val, m_idx] =
                            detail::argmin4_avx2_double(&priorities_[child_start]);
                        best_child_val = m_val;
                        best_child_offset = m_idx;
                    } else {
                        const std::size_t count = size_ - child_start;
                        auto [m_val, m_idx] =
                            detail::argmin4_scalar_branchless(&priorities_[child_start], count);
                        best_child_val = m_val;
                        best_child_offset = m_idx;
                    }
                } else if constexpr (std::is_same_v<KeyType, float>) {
                    if (child_start + 4 <= size_) {
                        auto [m_val, m_idx] = detail::argmin4_avx2_float(&priorities_[child_start]);
                        best_child_val = m_val;
                        best_child_offset = m_idx;
                    } else {
                        const std::size_t count = size_ - child_start;
                        auto [m_val, m_idx] =
                            detail::argmin4_scalar_branchless(&priorities_[child_start], count);
                        best_child_val = m_val;
                        best_child_offset = m_idx;
                    }
                } else {
                    const std::size_t count = std::min<std::size_t>(4, size_ - child_start);
                    auto [m_val, m_idx] =
                        detail::argmin4_scalar_branchless(&priorities_[child_start], count);
                    best_child_val = m_val;
                    best_child_offset = m_idx;
                }
#else
                const std::size_t count = std::min<std::size_t>(4, size_ - child_start);
                auto [m_val, m_idx] =
                    detail::argmin4_scalar_branchless(&priorities_[child_start], count);
                best_child_val = m_val;
                best_child_offset = m_idx;
#endif
            } else {
                const std::size_t count = std::min<std::size_t>(Arity, size_ - child_start);
                auto [m_val, m_idx] =
                    detail::argmin4_scalar_branchless(&priorities_[child_start], count);
                best_child_val = m_val;
                best_child_offset = m_idx;
            }

            if (val <= best_child_val) {
                break;
            }

            const std::size_t best_child_idx =
                child_start + static_cast<std::size_t>(best_child_offset);
            priorities_[hole] = best_child_val;
            vertices_[hole] = vertices_[best_child_idx];
            pos_in_heap_[static_cast<std::size_t>(vertices_[hole])] = static_cast<int32_t>(hole);
            hole = best_child_idx;
        }

        priorities_[hole] = val;
        vertices_[hole] = id;
        pos_in_heap_[static_cast<std::size_t>(id)] = static_cast<int32_t>(hole);
    }

    std::size_t size_{0};
    alignas(64) std::vector<KeyType, detail::AlignedAllocator<KeyType, 64>> priorities_;
    alignas(64) std::vector<IndexType, detail::AlignedAllocator<IndexType, 64>> vertices_;
    std::vector<int32_t> pos_in_heap_;
};

} // namespace algoat::graph
