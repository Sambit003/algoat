/**
 * @file countingsort.hpp
 * @brief Counting Sort implementation.
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
void countingsort_impl(T* arr, std::size_t n) {
    if (n <= 1) return;

    auto [min_it, max_it] = std::minmax_element(arr, arr + n);
    T min_val = *min_it;
    T max_val = *max_it;

    std::size_t range = static_cast<std::size_t>(max_val - min_val + 1);
    std::vector<std::size_t> count(range, 0);

    for (std::size_t i = 0; i < n; ++i) {
        count[static_cast<std::size_t>(arr[i] - min_val)]++;
    }

    std::size_t idx = 0;
    for (std::size_t i = 0; i < range; ++i) {
        while (count[i] > 0) {
            arr[idx++] = static_cast<T>(i + min_val);
            count[i]--;
        }
    }
}

} // namespace detail

template <typename T>
void countingsort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::countingsort_impl(data.data(), data.size());
}

struct CountingSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "countingsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        countingsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 1000;
    }
};

} // namespace algoat::sorting