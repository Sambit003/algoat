/**
 * @file radixsort.hpp
 * @brief Least Significant Digit (LSD) and Most Significant Digit (MSD) Radix Sort.
 *
 * Provides non-comparative linear-time sorting for integral types by processing
 * byte-by-byte (8-bit radix = 256 buckets). Signed integers are seamlessly supported
 * by flipping the most significant sign bit via XOR with <tt>1 << (sizeof(T)*8 - 1)</tt>.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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
    template <typename T>
        requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
    void sort(std::span<T> arr) const {
        if (arr.empty())
            return;

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
    template <typename T> static void msd_impl(std::span<T> arr, std::span<T> buffer, int shift) {
        if (arr.size() <= 1)
            return;

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
                    msd_impl<T>(arr.subspan(boundaries[i], bin_size),
                                buffer.subspan(boundaries[i], bin_size), shift - 8);
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
    template <typename T>
        requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
    void sort(std::span<T> arr) const {
        if (arr.size() <= 1)
            return;
        std::vector<T> buffer(arr.size());
        msd_impl<T>(arr, buffer, (sizeof(T) - 1) * 8);
    }
};

namespace detail {

template <typename K> constexpr auto to_sortable_key(K key) noexcept {
    using Raw = std::remove_cvref_t<K>;
    if constexpr (std::is_floating_point_v<Raw>) {
        if constexpr (sizeof(Raw) == 4) {
            uint32_t u = std::bit_cast<uint32_t>(key);
            constexpr uint32_t sign_mask = 1U << 31;
            return (u & sign_mask) ? ~u : (u ^ sign_mask);
        } else if constexpr (sizeof(Raw) == 8) {
            uint64_t u = std::bit_cast<uint64_t>(key);
            constexpr uint64_t sign_mask = 1ULL << 63;
            return (u & sign_mask) ? ~u : (u ^ sign_mask);
        } else {
            static_assert(sizeof(Raw) == 4 || sizeof(Raw) == 8,
                          "Unsupported floating-point width for in-place radix sort");
        }
    } else if constexpr (std::is_integral_v<Raw>) {
        using U = std::make_unsigned_t<Raw>;
        U u = static_cast<U>(key);
        if constexpr (std::is_signed_v<Raw>) {
            constexpr U sign_mask = U(1) << (sizeof(U) * 8 - 1);
            return static_cast<U>(u ^ sign_mask);
        } else {
            return u;
        }
    } else {
        return key;
    }
}

template <typename T, typename Projection>
void branchless_small_sort(std::span<T> data, Projection proj) {
    const std::size_t n = data.size();
    for (std::size_t i = 1; i < n; ++i) {
        T key = std::move(data[i]);
        auto key_val = to_sortable_key(std::invoke(proj, key));
        std::size_t dest = i;
        for (std::size_t j = i; j > 0; --j) {
            bool cmp = to_sortable_key(std::invoke(proj, data[j - 1])) > key_val;
            dest -= cmp;
        }
        for (std::size_t j = i; j > dest; --j) {
            data[j] = std::move(data[j - 1]);
        }
        data[dest] = std::move(key);
    }
}

template <typename T, typename Projection>
void inplace_radix_sort_impl(std::span<T> data, Projection proj, int shift) {
    while (true) {
        const std::size_t n = data.size();
        if (n <= 64) {
            branchless_small_sort(data, proj);
            return;
        }

        auto extract_byte = [&](const T& elem) -> uint8_t {
            auto key = to_sortable_key(std::invoke(proj, elem));
            return static_cast<uint8_t>((key >> shift) & 0xFF);
        };

        // 1. Compute 256-bucket histogram
        std::size_t bucket_end[256] = {0};
        for (const auto& elem : data) {
            bucket_end[extract_byte(elem)]++;
        }

        // Check if all elements share the same radix byte (adaptive common-prefix skipping)
        bool all_same = false;
        for (std::size_t i = 0; i < 256; ++i) {
            if (bucket_end[i] == n) {
                all_same = true;
                break;
            }
        }

        if (all_same) {
            if (shift >= 8) {
                shift -= 8;
                continue;
            }
            return;
        }

        // 2. Compute bucket boundaries via prefix sums
        std::size_t next_offset[256];
        std::size_t sum = 0;
        for (std::size_t i = 0; i < 256; ++i) {
            next_offset[i] = sum;
            sum += bucket_end[i];
            bucket_end[i] = sum;
        }

        // 3. In-situ cycle-following permutation (Ska Swap)
        for (std::size_t b = 0; b < 255; ++b) {
            while (next_offset[b] < bucket_end[b]) {
                if (extract_byte(data[next_offset[b]]) == b) {
                    ++next_offset[b];
                    continue;
                }

                T current = std::move(data[next_offset[b]]);
                uint8_t dest = extract_byte(current);

                while (dest != b) {
                    while (extract_byte(data[next_offset[dest]]) == dest) {
                        ++next_offset[dest];
                    }
                    std::swap(current, data[next_offset[dest]]);
                    ++next_offset[dest];
                    dest = extract_byte(current);
                }

                data[next_offset[b]] = std::move(current);
                ++next_offset[b];
            }
        }

        // 4. Recursive subdivision into sub-buckets
        if (shift >= 8) {
            std::size_t prev_end = 0;
            for (std::size_t i = 0; i < 256; ++i) {
                std::size_t curr_end = bucket_end[i];
                std::size_t bucket_size = curr_end - prev_end;
                if (bucket_size > 1) {
                    if (bucket_size <= 64) {
                        branchless_small_sort(data.subspan(prev_end, bucket_size), proj);
                    } else {
                        inplace_radix_sort_impl(data.subspan(prev_end, bucket_size), proj,
                                                shift - 8);
                    }
                }
                prev_end = curr_end;
            }
        }

        return;
    }
}

} // namespace detail

/**
 * @brief In-place Hybrid MSD Radix Sort using register-resident cycle chasing (Ska Sort).
 *
 * Partitions elements in-situ without auxiliary heap allocation (O(1) auxiliary heap space),
 * utilizing a small stack-allocated histogram and cycle-following permutation.
 *
 * @tparam T Element type.
 * @tparam Projection Projection callable returning a sortable integral or floating-point key.
 * @param data Contiguous span of elements to sort in-place.
 * @param proj Projection function applied to each element.
 */
template <typename T, typename Projection>
void inplace_radix_sort(std::span<T> data, Projection proj) {
    if (data.size() <= 1) {
        return;
    }
    using Key = decltype(detail::to_sortable_key(std::invoke(proj, std::declval<T&>())));
    constexpr std::size_t K = sizeof(Key);
    static_assert(K > 0, "Key type must have non-zero size");
    const int initial_shift = static_cast<int>((K - 1) * 8);
    detail::inplace_radix_sort_impl(data, proj, initial_shift);
}

/**
 * @brief In-place Hybrid MSD Radix Sort for integral types.
 *
 * @tparam T Integral type (excluding bool).
 * @param data Contiguous span of integers to sort in-place.
 */
template <std::integral T> void inplace_radix_sort(std::span<T> data) noexcept {
    inplace_radix_sort(data, std::identity{});
}

/**
 * @brief In-place Hybrid MSD Radix Sort for floating-point types (IEEE-754).
 *
 * @tparam T Floating-point type (float, double).
 * @param data Contiguous span of floats to sort in-place.
 */
template <std::floating_point T> void inplace_radix_sort(std::span<T> data) noexcept {
    inplace_radix_sort(data, std::identity{});
}

/**
 * @struct RadixSortInPlaceMSD
 * @brief In-Place Hybrid MSD Radix Sort (Ska Sort) with O(1) auxiliary heap memory.
 *
 * Partitions elements in-situ via cycle-following permutations (Ska Sort style),
 * requiring zero dynamic heap allocations and stack-bounded histogram buffers.
 * Seamlessly sorts full signed/unsigned integer ranges and IEEE-754 floating-point keys.
 */
struct RadixSortInPlaceMSD {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "radixsortinplacemsd"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "radixsortinplacemsd";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Sorts a span in-place using Hybrid In-Place MSD Radix Sort.
     * @tparam T Must satisfy integral or floating-point requirements (excluding bool).
     * @param arr Span of elements to sort in-place.
     */
    template <typename T>
        requires((std::is_integral_v<T> || std::is_floating_point_v<T>) && !std::is_same_v<T, bool>)
    void sort(std::span<T> arr) const {
        inplace_radix_sort(arr);
    }
};

} // namespace algoat::sorting

#include "algoat/core/registry.hpp"

ALGOAT_REGISTER_ALGORITHM("sorting", "radixsortlsd", ::algoat::sorting::RadixSortLSD)
ALGOAT_REGISTER_ALGORITHM("sorting", "radixsortmsd", ::algoat::sorting::RadixSortMSD)
ALGOAT_REGISTER_ALGORITHM("sorting", "radixsortinplacemsd", ::algoat::sorting::RadixSortInPlaceMSD)
