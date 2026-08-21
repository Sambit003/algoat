/**
 * @file countingsort.hpp
 * @brief Stable non-comparative Counting Sort implementation.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace algoat::sorting {

/**
 * @struct CountingSort
 * @brief Stable non-comparative integer sorting algorithm.
 *
 * Computes frequency histogram of keys, calculates cumulative prefix sums,
 * and distributes elements into their stable positions.
 *
 * @par Characteristics:
 * - <b>Category:</b> Non-comparative, Integer Key Distribution.
 * - <b>Stability:</b> Stable.
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N + K) auxiliary buffer
 */
struct CountingSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "countingsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "countingsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts an integral span using stable Counting Sort.
     * @tparam T Must satisfy <tt>std::is_integral_v<T></tt>.
     *
     * @param arr Contiguous span of integers to sort.
     * @throws std::invalid_argument If @c T is non-integral.
     */
    template <typename T>
        requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
    void sort(std::span<T> arr) const {
        if (arr.empty())
            return;

        auto [min_it, max_it] = std::minmax_element(arr.begin(), arr.end());
        T min_val = *min_it;
        T max_val = *max_it;

        std::size_t range =
            static_cast<std::size_t>(max_val) - static_cast<std::size_t>(min_val) + 1;

        std::vector<std::size_t> count(range, 0);
        for (T x : arr) {
            count[static_cast<std::size_t>(x) - static_cast<std::size_t>(min_val)]++;
        }

        // Cumulative prefix sums
        for (std::size_t i = 1; i < range; ++i) {
            count[i] += count[i - 1];
        }

        // Place elements in reverse to preserve stability
        std::vector<T> output(arr.size());
        for (std::size_t i = arr.size(); i-- > 0;) {
            std::size_t diff = static_cast<std::size_t>(arr[i]) - static_cast<std::size_t>(min_val);
            output[count[diff] - 1] = arr[i];
            count[diff]--;
        }

        std::copy(output.begin(), output.end(), arr.begin());
    }
};

} // namespace algoat::sorting
