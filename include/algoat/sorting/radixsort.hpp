/**
 * @file radixsort.hpp
 * @brief Radix Sort implementation (LSD and MSD).
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>
#include <algorithm>

namespace algoat::sorting {

namespace detail {

template <typename T>
void count_sort_lsd(T* arr, std::size_t n, std::size_t exp) {
    std::vector<T> output(n);
    std::vector<std::size_t> count(10, 0);

    for (std::size_t i = 0; i < n; ++i) {
        count[(arr[i] / exp) % 10]++;
    }

    for (std::size_t i = 1; i < 10; ++i) {
        count[i] += count[i - 1];
    }

    for (std::size_t i = n; i > 0; --i) {
        std::size_t idx = (arr[i - 1] / exp) % 10;
        output[count[idx] - 1] = arr[i - 1];
        count[idx]--;
    }

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = output[i];
    }
}

} // namespace detail

template <typename T>
void radixsort_lsd(std::span<T> data) {
    if (data.size() <= 1) return;

    T max_val = *std::max_element(data.begin(), data.end());
    for (std::size_t exp = 1; max_val / exp > 0; exp *= 10) {
        detail::count_sort_lsd(data.data(), data.size(), exp);
    }
}

template <typename T>
void radixsort_msd(std::span<T> data) {
    // MSD implementation (simplified: call LSD for now)
    radixsort_lsd(data);
}

struct RadixSortLSD {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "radixsort_lsd";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        radixsort_lsd(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 1000;
    }
};

struct RadixSortMSD {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "radixsort_msd";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        radixsort_msd(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 1000;
    }
};

} // namespace algoat::sorting