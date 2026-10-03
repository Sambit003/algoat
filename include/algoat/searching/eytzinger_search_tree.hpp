/**
 * @file eytzinger_search_tree.hpp
 * @brief Cache-Oblivious Eytzinger (Breadth-First / Complete Binary Heap) Search Engine.
 */

#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif

namespace algoat::searching {

namespace detail {

/**
 * @brief Software memory prefetch instruction wrapper for GCC, Clang, and MSVC.
 */
template <typename Ptr> inline void prefetch_read(const Ptr* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(static_cast<const void*>(ptr), 0, 0);
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(static_cast<const void*>(ptr)), _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

} // namespace detail

/**
 * @class EytzingerSearchTree
 * @brief Cache-oblivious binary search tree using breadth-first complete binary tree array layout.
 *
 * Rearranges sorted data into 1-indexed breadth-first Eytzinger order. Top levels reside
 * contiguously in L1 cache lines, and deterministic software lookahead prefetching overlaps
 * DRAM memory access latency with ALU instructions.
 *
 * @tparam T Element type satisfying std::totally_ordered.
 */
template <std::totally_ordered T> class EytzingerSearchTree {
public:
    /**
     * @brief Constructs an EytzingerSearchTree from a sorted span of data in O(N) time.
     * @param sorted_data Range of elements sorted in non-decreasing order.
     */
    explicit EytzingerSearchTree(std::span<const T> sorted_data) {
        const std::size_t n = sorted_data.size();
        tree_.resize(n + 1);
        eytzinger_to_sorted_.resize(n + 1);
        if (n > 0) {
            std::size_t sorted_idx = 0;
            build_tree(1, sorted_data, sorted_idx);
        }
    }

    /**
     * @brief Performs a branchless lower bound search query with software lookahead prefetching.
     * @param target Search key value.
     * @return 0-based index in the original sorted data of the first element >= target,
     *         or size() if all elements are < target.
     */
    [[nodiscard]] std::size_t lower_bound(const T& target) const noexcept {
        if (size() == 0) {
            return 0;
        }

        std::size_t k = 1;
        const std::size_t n = size();
        constexpr std::size_t line_elements = (sizeof(T) <= 64) ? (64 / sizeof(T)) : 1;

        while (k <= n) {
            std::size_t prefetch_idx = (k * line_elements < n) ? (k * line_elements) : n;
            detail::prefetch_read(&tree_[prefetch_idx]);
            k = (2 * k) + static_cast<std::size_t>(tree_[k] < target);
        }

        k >>= (std::countr_zero(~k) + 1);

        if (k == 0) {
            return n;
        }
        return eytzinger_to_sorted_[k];
    }

    /**
     * @brief Direct 1-indexed accessor into the internal Eytzinger breadth-first array.
     * @param eytzinger_idx 1-based index (1 <= eytzinger_idx <= size()).
     * @return Reference to the element at the specified Eytzinger array position.
     */
    [[nodiscard]] const T& operator[](std::size_t eytzinger_idx) const noexcept {
        return tree_[eytzinger_idx];
    }

    /**
     * @brief Returns the total number of elements in the search tree.
     * @return Number of elements stored.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return tree_.empty() ? 0 : tree_.size() - 1;
    }

private:
    std::vector<T> tree_;
    std::vector<std::size_t> eytzinger_to_sorted_;

    void build_tree(std::size_t k, std::span<const T> sorted_data, std::size_t& sorted_idx) {
        if (k <= sorted_data.size()) {
            build_tree(2 * k, sorted_data, sorted_idx);
            tree_[k] = sorted_data[sorted_idx];
            eytzinger_to_sorted_[k] = sorted_idx;
            ++sorted_idx;
            build_tree(2 * k + 1, sorted_data, sorted_idx);
        }
    }
};

} // namespace algoat::searching
