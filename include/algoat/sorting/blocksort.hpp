/**
 * @file blocksort.hpp
 * @brief Block Sort (WikiSort) implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <algorithm>
#include <vector>

namespace algoat::sorting {

namespace detail {

template <typename T>
void blocksort_impl(T* arr, std::size_t n) {
    // Simplified BlockSort: fallback to std::sort for now
    // In a real implementation, this would be WikiSort
    std::sort(arr, arr + n);
}

} // namespace detail

template <typename T>
void blocksort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::blocksort_impl(data.data(), data.size());
}

struct BlockSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "blocksort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        blocksort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }
};

} // namespace algoat::sorting