/**
 * @file bucketsort.hpp
 * @brief Bucket Sort implementation.
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
void bucketsort_impl(T* arr, std::size_t n) {
    if (n <= 1) return;

    auto [min_it, max_it] = std::minmax_element(arr, arr + n);
    T min_val = *min_it;
    T max_val = *max_it;

    if (min_val == max_val) return;

    std::size_t num_buckets = n;
    std::vector<std::vector<T>> buckets(num_buckets);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t idx = static_cast<std::size_t>(
            (static_cast<double>(arr[i] - min_val) / (max_val - min_val)) * (num_buckets - 1)
        );
        buckets[idx].push_back(arr[i]);
    }

    std::size_t idx = 0;
    for (auto& bucket : buckets) {
        std::sort(bucket.begin(), bucket.end());
        for (const auto& val : bucket) {
            arr[idx++] = val;
        }
    }
}

} // namespace detail

template <typename T>
void bucketsort(std::span<T> data) {
    if (data.size() <= 1) return;
    detail::bucketsort_impl(data.data(), data.size());
}

struct BucketSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "bucketsort";
    }

    template <typename T>
    void sort(std::span<T> data) const {
        bucketsort(data);
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 1000;
    }
};

} // namespace algoat::sorting