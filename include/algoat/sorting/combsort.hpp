/**
 * @file combsort.hpp
 * @brief Comb Sort implementation.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail {

template <typename T>
void combsort_impl(T* arr, std::size_t n) {
    std::size_t gap = n;
    bool swapped = true;

    while (gap > 1 || swapped) {
        if (gap > 1) {
            gap = static_cast<std::size_t>(gap / 1.3);
        }
        swapped = false;
        for (std::size_t i = 0; i + gap < n; ++i) {
            if (arr[i] > arr[i + gap]) {
                std::swap(arr[i], arr[i + gap]);
                swapped = true;
            }
        }
    }
}

} // namespace detail

template <typename T>
void combsort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::combsort_impl(data.data(), data.size());
}

struct CombSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "combsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        combsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 16;
    }
};

} // namespace algoat::sorting