/**
 * @file float16_sort.hpp
 * @brief High-performance non-comparative sorting for IEEE-754 16-bit floating-point numbers.
 * 
 * Implements an O(N) integer-transformation technique that converts IEEE-754 float16
 * bit patterns into strictly order-preserving 16-bit unsigned integers:
 * 
 *
 * @par Bit-Flipping Transformation:
 * IEEE-754 floating point uses sign-magnitude representation:
 * - Positive floats: 0 | Exponent | Mantissa (monotonic with integer order)
 * - Negative floats: 1 | Exponent | Mantissa (reversed magnitude relative to integer order)
 * 
 * To map floats to an ordered @c uint16_t domain:
 * - If negative (<tt>u & 0x8000 != 0</tt>): invert all bits (@c ~u), converting @c -inf to lowest unsigned values.
 * - If positive (<tt>u & 0x8000 == 0</tt>): flip the sign bit (<tt>u | 0x8000</tt>), placing them above negative numbers.
 * 
 * @code{.text}
 *   Original Float16 Value -> Transformed uint16_t Key
 *   -Infinity               -> 0x0000 (smallest)
 *   -1.0                    -> 0x43FF
 *   -0.0                    -> 0x7FFF
 *   +0.0                    -> 0x8000
 *   +1.0                    -> 0xBC00
 *   +Infinity               -> 0xFFFF (largest)
 * @endcode
 * 
 *
 * @par Multi-Tier Sorting Strategy:
 * 1. **Small arrays (<tt>N < 256</tt>)**: @c std::sort with @c Float16Compare transparent comparator.
 * 2. **Medium arrays (<tt>256 <= N < 65,536</tt>)**: 2-pass 8-bit Radix Sort (L1/L2 cache friendly with 256-entry histograms).
 * 3. **Large arrays (<tt>N >= 65,536</tt>)**: Single-pass 16-bit Counting Sort (65,536 buckets in <tt>O(N)</tt>).
 */

#pragma once

#include <cstdint>
#include <span>
#include <algorithm>
#include <concepts>
#include <type_traits>
#include <cstring>
#include <vector>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace algoat::concepts {

/**
 * @concept Size16Bit
 * @brief Matches any type with a 2-byte (16-bit) memory footprint.
 */
template <typename T>
concept Size16Bit = sizeof(T) == 2;

/**
 * @concept Int16
 * @brief Matches 16-bit integral types (e.g., int16_t, uint16_t) excluding bool.
 */
template <typename T>
concept Int16 = std::integral<T> && Size16Bit<T> && !std::same_as<std::remove_cv_t<T>, bool>;

/**
 * @concept Float16
 * @brief Matches 16-bit floating point types (e.g., std::float16_t).
 */
template <typename T>
concept Float16 = std::floating_point<T> && Size16Bit<T>;

/**
 * @concept Float16OrInt16
 * @brief Matches either 16-bit float or 16-bit integral types.
 */
template <typename T>
concept Float16OrInt16 = Int16<T> || Float16<T>;

} // namespace algoat::concepts

namespace algoat::numerics {

/**
 * @struct Float16Compare
 * @brief Transparent comparator using the IEEE-754 bit-flip order transformation.
 * 
 * Enables branchless comparison of 16-bit floating point representations
 * without floating-point ALU instructions.
 */
struct Float16Compare {
    using is_transparent = void;

    template <algoat::concepts::Float16OrInt16 T, algoat::concepts::Float16OrInt16 U>
    constexpr bool operator()(const T& a, const U& b) const noexcept {
        auto to_ordered = [](auto x) -> uint16_t {
            uint16_t u;
            std::memcpy(&u, &x, 2);
            return (u & 0x8000) ? ~u : (u | 0x8000);
        };
        return to_ordered(a) < to_ordered(b);
    }
};

/**
 * @brief High-performance multi-tier sort for 16-bit float and integer sequences.
 * 
 *
 * @par Time Complexity: <tt>O(N)</tt> for <tt>N >= 256</tt>; <tt>O(N log N)</tt> for <tt>N < 256</tt>.
 *
 * @par Space Complexity: <tt>O(N)</tt> auxiliary buffer for radix/counting passes.
 * 
 * @tparam T 16-bit numeric type satisfying @c concepts::Float16OrInt16.
 *
 * @param data Contiguous span of 16-bit elements to sort in-place.
 */
template <algoat::concepts::Float16OrInt16 T>
void sort_float16(std::span<T> data) {
    if (data.size() < 256) {
        std::sort(data.begin(), data.end(), Float16Compare{});
        return;
    }
    
    if (data.size() < 65536) {
        // 2-pass 8-bit Radix Sort for smaller arrays (less memset overhead)
        std::vector<T> buffer(data.size());
        std::span<T> src = data;
        std::span<T> dst = buffer;
        
        for (int shift = 0; shift < 16; shift += 8) {
            std::size_t count[256] = {0};
            
            for (T val : src) {
                uint16_t u;
                std::memcpy(&u, &val, 2);
                uint16_t ordered = (u & 0x8000) ? ~u : (u | 0x8000);
                count[(ordered >> shift) & 0xFF]++;
            }
            
            std::size_t total = 0;
            for (int i = 0; i < 256; ++i) {
                std::size_t oldCount = count[i];
                count[i] = total;
                total += oldCount;
            }
            
            for (T val : src) {
                uint16_t u;
                std::memcpy(&u, &val, 2);
                uint16_t ordered = (u & 0x8000) ? ~u : (u | 0x8000);
                std::size_t bucket = (ordered >> shift) & 0xFF;
                dst[count[bucket]++] = val;
            }
            
            std::swap(src, dst);
        }
    } else {
        // Single-pass 16-bit counting sort (65536 buckets)
        std::size_t count[65536] = {0};
        
        for (T val : data) {
            uint16_t u;
            std::memcpy(&u, &val, 2);
            uint16_t ordered = (u & 0x8000) ? ~u : (u | 0x8000);
            count[ordered]++;
        }
        
        std::size_t total = 0;
        for (int i = 0; i < 65536; ++i) {
            std::size_t oldCount = count[i];
            count[i] = total;
            total += oldCount;
        }
        
        std::vector<T> buffer(data.size());
        for (T val : data) {
            uint16_t u;
            std::memcpy(&u, &val, 2);
            uint16_t ordered = (u & 0x8000) ? ~u : (u | 0x8000);
            buffer[count[ordered]++] = val;
        }
        
        std::copy(buffer.begin(), buffer.end(), data.begin());
    }
}

} // namespace algoat::numerics
