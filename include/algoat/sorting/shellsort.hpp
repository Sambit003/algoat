/**
 * @file shellsort.hpp
 * @brief Shell Sort implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail {

template <typename T>
void shellsort_impl(T* arr, std::size_t n) {
    for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
        for (std::size_t i = gap; i < n; ++i) {
            T temp = std::move(arr[i]);
            std::size_t j = i;
            while (j >= gap && arr[j - gap] > temp) {
                arr[j] = std::move(arr[j - gap]);
                j -= gap;
            }
            arr[j] = std::move(temp);
        }
    }
}

} // namespace detail

template <typename T>
void shellsort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::shellsort_impl(data.data(), data.size());
}

struct ShellSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "shellsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        shellsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 16;
    }
};

} // namespace algoat::sorting