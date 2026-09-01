/**
 * @file insertionsort.hpp
 * @brief Insertion Sort implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail {

template <typename T>
void insertionsort_impl(T* arr, std::size_t n) {
    for (std::size_t i = 1; i < n; ++i) {
        T key = std::move(arr[i]);
        std::size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = std::move(arr[j - 1]);
            --j;
        }
        arr[j] = std::move(key);
    }
}

} // namespace detail

/**
 * @brief Sorts the span using Insertion Sort.
 * @tparam T Element type.
 * @param data Span of elements to sort.
 */
template <typename T>
void insertionsort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::insertionsort_impl(data.data(), data.size());
}

struct InsertionSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "insertionsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        insertionsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }
};

} // namespace algoat::sorting