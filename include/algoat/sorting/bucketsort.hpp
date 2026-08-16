/**
 * @file bucketsort.hpp
 * @brief Distribution-based Bucket Sort algorithm implementation.
 */

#pragma once

#include <string_view>
#include <span>
#include <concepts>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

namespace algoat::sorting {

/**
 * @struct BucketSort
 * @brief Distribution sorting algorithm partitioning elements across sub-buckets.
 * 
 * Divides the input range uniformly across ~N / 10 buckets, sorts each
 * bucket with @c std::sort, and concatenates the results.
 *
 * @par Characteristics:
 * - <b>Category:</b> Distribution.
 * - <b>Stability:</b> Dependent on bucket sort stability.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N + K) (uniform distribution)
 * - Average Case: @c O(N + K)
 * - Worst Case: @c O(N^2) (all elements cluster in a single bucket)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N + K) auxiliary space for buckets
 */
struct BucketSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "bucketsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "bucketsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts an integral span using Bucket Sort.
     * @tparam T Must satisfy <tt>std::is_integral_v<T></tt>.
 *
 * @param arr Contiguous span of integers to sort.
     * @throws std::invalid_argument If @c T is non-integral.
     */
    template<typename T>
    void sort(std::span<T> arr) const {
        if constexpr (!std::is_integral_v<T>) {
            throw std::invalid_argument("BucketSort requires an integral type");
        } else {
            if (arr.empty()) return;

            auto [min_it, max_it] = std::minmax_element(arr.begin(), arr.end());
            T min_val = *min_it;
            
            using U = std::make_unsigned_t<T>;
            U range = static_cast<U>(*max_it - min_val);
            
            if (range == 0) return;

            std::size_t num_buckets = std::max<std::size_t>(1, arr.size() / 10);
            std::vector<std::vector<T>> buckets(num_buckets);
            
            for (T x : arr) {
                std::size_t b_idx = static_cast<std::size_t>((static_cast<double>(static_cast<U>(x - min_val)) / range) * (num_buckets - 1));
                buckets[b_idx].push_back(x);
            }
            
            std::size_t idx = 0;
            for (auto& bucket : buckets) {
                std::sort(bucket.begin(), bucket.end());
                for (T x : bucket) {
                    arr[idx++] = x;
                }
            }
        }
    }
};

} // namespace algoat::sorting
