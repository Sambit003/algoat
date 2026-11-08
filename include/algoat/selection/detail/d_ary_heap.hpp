/**
 * @file d_ary_heap.hpp
 * @brief Cache-conscious contiguous generic d-ary heap container.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace algoat::selection::detail {

/**
 * @class d_ary_heap
 * @brief High-performance contiguous d-ary priority queue.
 *
 * Sibling nodes are laid out contiguously in memory. For Arity=4, 4 sibling keys fit
 * entirely within a single 64-byte L1 cache line, halving tree height and minimizing DRAM latency.
 *
 * @tparam T Value type.
 * @tparam Arity Branching factor (default 4).
 * @tparam Comp Strict weak ordering comparator (default std::less<T> for max-heap).
 */
template <typename T, std::size_t Arity = 4, typename Comp = std::less<T>> class d_ary_heap {
    static_assert(Arity >= 2, "d_ary_heap arity must be at least 2");

public:
    using value_type = T;
    using size_type = std::size_t;
    using compare_type = Comp;

    explicit d_ary_heap(Comp comp = Comp{}) : comp_(std::move(comp)) {}

    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }

    [[nodiscard]] size_type size() const noexcept {
        return data_.size();
    }

    [[nodiscard]] const T& top() const {
        if (data_.empty()) {
            throw std::underflow_error("top() called on empty d_ary_heap");
        }
        return data_.front();
    }

    void reserve(size_type capacity) {
        data_.reserve(capacity);
    }

    void clear() noexcept {
        data_.clear();
    }

    void push(const T& value) {
        data_.push_back(value);
        sift_up(data_.size() - 1);
    }

    void push(T&& value) {
        data_.push_back(std::move(value));
        sift_up(data_.size() - 1);
    }

    void pop() {
        if (data_.empty()) {
            throw std::underflow_error("pop() called on empty d_ary_heap");
        }
        data_.front() = std::move(data_.back());
        data_.pop_back();
        if (!data_.empty()) {
            sift_down(0);
        }
    }

private:
    std::vector<T> data_;
    [[no_unique_address]] Comp comp_{};

    static constexpr size_type parent_of(size_type idx) noexcept {
        if constexpr (Arity == 4) {
            return (idx - 1) >> 2;
        } else {
            return (idx - 1) / Arity;
        }
    }

    static constexpr size_type first_child_of(size_type idx) noexcept {
        if constexpr (Arity == 4) {
            return (idx << 2) + 1;
        } else {
            return Arity * idx + 1;
        }
    }

    void sift_up(size_type idx) {
        while (idx > 0) {
            const size_type p = parent_of(idx);
            if (comp_(data_[p], data_[idx])) {
                using std::swap;
                swap(data_[p], data_[idx]);
                idx = p;
            } else {
                break;
            }
        }
    }

    void sift_down(size_type idx) {
        const size_type n = data_.size();
        while (true) {
            const size_type first_child = first_child_of(idx);
            if (first_child >= n) {
                break;
            }

            size_type best_child = first_child;
            const size_type last_child = std::min(first_child + Arity, n);
            for (size_type c = first_child + 1; c < last_child; ++c) {
                if (comp_(data_[best_child], data_[c])) {
                    best_child = c;
                }
            }

            if (comp_(data_[idx], data_[best_child])) {
                using std::swap;
                swap(data_[idx], data_[best_child]);
                idx = best_child;
            } else {
                break;
            }
        }
    }
};

} // namespace algoat::selection::detail
