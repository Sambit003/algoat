/**
 * @file introsort.hpp
 * @brief IntroSort implementation (QuickSort + HeapSort fallback).
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>
#include <algorithm>
#include <cmath>

namespace algoat::sorting {

namespace detail {

template <typename T>
void introsort_impl(T* arr, std::size_t low, std::size_t high, std::size_t depth_limit) {
    if (high - low <= 16) {
        // Use InsertionSort for small arrays
        for (std::size_t i = low + 1; i <= high; ++i) {
            T key = std::move(arr[i]);
            std::size_t j = i;
            while (j > low && arr[j - 1] > key) {
                arr[j] = std::move(arr[j - 1]);
                --j;
            }
            arr[j] = std::move(key);
        }
        return;
    }

    if (depth_limit == 0) {
        // Use HeapSort if depth limit reached
        std::make_heap(arr + low, arr + high + 1);
        std::sort_heap(arr + low, arr + high + 1);
        return;
    }

    // QuickSort partition
    std::size_t pivot = arr[high];
    std::size_t i = low;
    for (std::size_t j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            std::swap(arr[i], arr[j]);
            ++i;
        }
    }
    std::swap(arr[i], arr[high]);

    introsort_impl(arr, low, i > 0 ? i - 1 : 0, depth_limit - 1);
    introsort_impl(arr, i + 1, high, depth_limit - 1);
}

} // namespace detail

template <typename T>
void introsort(std::span<T> data) {
    if (data.size() <= 1) return;
    std::size_t depth_limit = 2 * static_cast<std::size_t>(std::log2(data.size()));
    detail::introsort_impl(data.data(), 0, data.size() - 1, depth_limit);
}

struct IntroSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "introsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        introsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 16;
    }
};

} // namespace algoat::sorting