/**
 * @file timsort.hpp
 * @brief TimSort implementation.
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
void timsort_impl(T* arr, std::size_t n) {
    if (n <= 1) return;

    const std::size_t RUN = 32;

    // Sort individual runs using InsertionSort
    for (std::size_t i = 0; i < n; i += RUN) {
        std::size_t end = std::min(i + RUN, n);
        for (std::size_t j = i + 1; j < end; ++j) {
            T key = std::move(arr[j]);
            std::size_t k = j;
            while (k > i && arr[k - 1] > key) {
                arr[k] = std::move(arr[k - 1]);
                --k;
            }
            arr[k] = std::move(key);
        }
    }

    // Merge runs
    std::vector<T> temp(n);
    for (std::size_t size = RUN; size < n; size *= 2) {
        for (std::size_t left = 0; left < n; left += 2 * size) {
            std::size_t mid = std::min(left + size, n);
            std::size_t right = std::min(left + 2 * size, n);

            if (mid < right) {
                std::size_t i = left, j = mid, k = left;
                while (i < mid && j < right) {
                    if (arr[i] <= arr[j]) temp[k++] = std::move(arr[i++]);
                    else temp[k++] = std::move(arr[j++]);
                }
                while (i < mid) temp[k++] = std::move(arr[i++]);
                while (j < right) temp[k++] = std::move(arr[j++]);
                for (std::size_t p = left; p < right; ++p) {
                    arr[p] = std::move(temp[p]);
                }
            }
        }
    }
}

} // namespace detail

template <typename T>
void timsort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::timsort_impl(data.data(), data.size());
}

struct TimSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "timsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        timsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 32;
    }
};

} // namespace algoat::sorting