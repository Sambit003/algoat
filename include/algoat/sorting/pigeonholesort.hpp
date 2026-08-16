/**
 * @file pigeonholesort.hpp
 * @brief Distribution-based Pigeonhole Sort implementation.
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
 * @struct PigeonholeSort
 * @brief Non-comparative sorting algorithm for integers where range is approximately equal to length.
 * 
 * Sets up an array of "pigeonholes" indexed by <tt>(x - min_val)</tt> and populates them by frequency,
 * then writes values sequentially back into the array.
 *
 * @par Characteristics:
 * - <b>Category:</b> Non-comparative, Distribution.
 * - <b>Stability:</b> Stable.
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(K) auxiliary space for holes
 */
struct PigeonholeSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "pigeonholesort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "pigeonholesort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts an integral span using Pigeonhole Sort.
     * @tparam T Must satisfy <tt>std::is_integral_v<T></tt>.
 *
 * @param arr Contiguous span of integers to sort.
     * @throws std::invalid_argument If @c T is non-integral.
     */
    template<typename T>
    void sort(std::span<T> arr) const {
        if constexpr (!std::is_integral_v<T>) {
            throw std::invalid_argument("PigeonholeSort requires an integral type");
        } else {
            if (arr.empty()) return;

            auto [min_it, max_it] = std::minmax_element(arr.begin(), arr.end());
            T min_val = *min_it;
            
            using U = std::make_unsigned_t<T>;
            U range = static_cast<U>(*max_it - min_val) + 1;
            
            std::vector<std::size_t> holes(range, 0);
            for (T x : arr) {
                holes[static_cast<U>(x - min_val)]++;
            }
            
            std::size_t idx = 0;
            for (U i = 0; i < range; ++i) {
                while (holes[i] > 0) {
                    arr[idx++] = static_cast<T>(i) + min_val;
                    holes[i]--;
                }
            }
        }
    }
};

} // namespace algoat::sorting
