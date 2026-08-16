/**
 * @file radixsort.hpp
 * @brief Least Significant Digit (LSD) and Most Significant Digit (MSD) Radix Sort.
 * 
 * Provides non-comparative linear-time sorting for integral types by processing
 * byte-by-byte (8-bit radix = 256 buckets). Signed integers are seamlessly supported
 * by flipping the most significant sign bit via XOR with <tt>1 << (sizeof(T)*8 - 1)</tt>.
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
 * @struct RadixSortLSD
 * @brief Stable Least Significant Digit (LSD) Radix Sort for integers.
 * 
 * Iterates through digits from least significant byte (LSB) to most significant byte (MSB),
 * maintaining stability across <tt>sizeof(T)</tt> passes.
 *
 * @par Characteristics:
 * - <b>Category:</b> Non-comparative, Distribution.
 * - <b>Stability:</b> Stable.
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N) auxiliary buffer
 */
struct RadixSortLSD {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "radixsortlsd"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "radixsortlsd";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts an integral span using LSD Radix Sort.
     * @tparam T Must satisfy <tt>std::is_integral_v<T></tt>.
 *
 * @param arr Span of integers to sort in-place.
     * @throws std::invalid_argument If @c T is non-integral.
     */
    template<typename T>
    void sort(std::span<T> arr) const {
        if constexpr (!std::is_integral_v<T>) {
            throw std::invalid_argument("RadixSortLSD requires an integral type");
        } else {
            if (arr.empty()) return;

            using U = std::make_unsigned_t<T>;
            const int passes = sizeof(T);
            std::vector<T> buffer(arr.size());
            std::span<T> src = arr;
            std::span<T> dst = buffer;

            for (int shift = 0; shift < passes * 8; shift += 8) {
                std::size_t count[256] = {0};

                for (T val : src) {
                    U u_val = static_cast<U>(val);
                    if constexpr (std::is_signed_v<T>) {
                        u_val ^= (U(1) << (sizeof(T) * 8 - 1));
                    }
                    count[(u_val >> shift) & 0xFF]++;
                }

                std::size_t total = 0;
                for (int i = 0; i < 256; ++i) {
                    std::size_t oldCount = count[i];
                    count[i] = total;
                    total += oldCount;
                }

                for (T val : src) {
                    U u_val = static_cast<U>(val);
                    if constexpr (std::is_signed_v<T>) {
                        u_val ^= (U(1) << (sizeof(T) * 8 - 1));
                    }
                    std::size_t bucket = (u_val >> shift) & 0xFF;
                    dst[count[bucket]++] = val;
                }

                std::swap(src, dst);
            }

            if (passes % 2 != 0) {
                std::copy(buffer.begin(), buffer.end(), arr.begin());
            }
        }
    }
};

/**
 * @struct RadixSortMSD
 * @brief Recursive Most Significant Digit (MSD) Radix Sort for integers.
 * 
 * Partitions elements into 256 sub-buckets starting from MSB and recurses down to LSB.
 *
 * @par Characteristics:
 * - <b>Category:</b> Non-comparative, Distribution / Partition.
 * - <b>Stability:</b> Stable.
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N + k \cdot 256) auxiliary space
 */
struct RadixSortMSD {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "radixsortmsd"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "radixsortmsd";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Recursive MSD radix sort worker on sub-buckets.
     */
    template<typename T>
    static void msd_impl(std::span<T> arr, std::span<T> buffer, int shift) {
        if (arr.size() <= 1) return;

        using U = std::make_unsigned_t<T>;
        std::size_t count[256] = {0};

        for (T val : arr) {
            U u_val = static_cast<U>(val);
            if constexpr (std::is_signed_v<T>) {
                u_val ^= (U(1) << (sizeof(T) * 8 - 1));
            }
            count[(u_val >> shift) & 0xFF]++;
        }

        std::size_t boundaries[256];
        std::size_t total = 0;
        for (int i = 0; i < 256; ++i) {
            boundaries[i] = total;
            total += count[i];
        }

        std::size_t offsets[256];
        std::copy(std::begin(boundaries), std::end(boundaries), std::begin(offsets));
        
        for (T val : arr) {
            U u_val = static_cast<U>(val);
            if constexpr (std::is_signed_v<T>) {
                u_val ^= (U(1) << (sizeof(T) * 8 - 1));
            }
            std::size_t bucket = (u_val >> shift) & 0xFF;
            buffer[offsets[bucket]++] = val;
        }

        std::copy(buffer.begin(), buffer.begin() + arr.size(), arr.begin());

        if (shift > 0) {
            for (int i = 0; i < 256; ++i) {
                std::size_t bin_size = count[i];
                if (bin_size > 1) {
                    msd_impl<T>(
                        arr.subspan(boundaries[i], bin_size),
                        buffer.subspan(boundaries[i], bin_size),
                        shift - 8
                    );
                }
            }
        }
    }

    /**
     * @brief Sorts an integral span using recursive MSD Radix Sort.
     * @tparam T Must satisfy <tt>std::is_integral_v<T></tt>.
 *
 * @param arr Span of integers to sort in-place.
     * @throws std::invalid_argument If @c T is non-integral.
     */
    template<typename T>
    void sort(std::span<T> arr) const {
        if constexpr (!std::is_integral_v<T>) {
            throw std::invalid_argument("RadixSortMSD requires an integral type");
        } else {
            if (arr.size() <= 1) return;
            std::vector<T> buffer(arr.size());
            msd_impl<T>(arr, buffer, (sizeof(T) - 1) * 8);
        }
    }
};

} // namespace algoat::sorting
