/**
 * @file morton.hpp
 * @brief 2D Morton Z-Order spatial curve sorting for complex numbers.
 * 
 * Replaces standard 1D lexicographical complex sorting with a 2D spatial Morton (Z-order)
 * curve ordering. This guarantees that numbers close to each other in the 2D complex plane
 * (real and imaginary components) remain spatially adjacent in memory, significantly
 * improving cache locality and nearest-neighbor query performance.
 * 
 *
 * @par Morton Interleaving:
 * 32-bit real (@c x) and imaginary (@c y) floating point coordinates are mapped
 * to order-preserving unsigned integers and bit-interleaved into a single 64-bit key:
 * @code{.text}
 * Key: [y31, x31, y30, x30, ..., y0, x0]
 * @endcode
 * - On x86-64 CPUs with BMI2 support, bit deposition uses the hardware @c _pdep_u64 instruction.
 * - On non-BMI2 architectures, an optimized 5-step bitwise dilatation (@c split_by_1) is used.
 * 
 *
 * @par Sorting Strategy:
 * - For <tt>N < 256</tt>: @c std::sort with @c MortonCompare transparent comparator.
 * - For <tt>N >= 256</tt>: 4-pass 16-bit Radix Sort across 64-bit keys in <tt>O(N)</tt> time.
 */

#pragma once

#include <complex>
#include <tuple>
#include <cstdint>
#include <bit>
#include <cstring>
#include <vector>
#include <span>
#include <algorithm>

#if defined(__BMI2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace algoat::numerics {

/**
 * @brief Interleaves two 32-bit unsigned integers into a 64-bit Morton Z-order key.
 * 
 * Places bits of @c x at even bit positions (0, 2, 4, ...) and bits of @c y at odd
 * bit positions (1, 3, 5, ...).
 * 
 *
 * @param x 32-bit unsigned integer for the X dimension (even bits).
 *
 * @param y 32-bit unsigned integer for the Y dimension (odd bits).
 * @return @c uint64_t 64-bit interleaved Morton key.
 */
inline uint64_t morton_interleave(uint32_t x, uint32_t y) noexcept {
#if defined(__BMI2__)
    return _pdep_u64(x, 0x5555555555555555ULL) | 
           _pdep_u64(y, 0xAAAAAAAAAAAAAAAAULL);
#else
    auto split_by_1 = [](uint32_t a) -> uint64_t {
        uint64_t x = a & 0xFFFFFFFFULL;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
        x = (x | (x << 2))  & 0x3333333333333333ULL;
        x = (x | (x << 1))  & 0x5555555555555555ULL;
        return x;
    };
    return split_by_1(x) | (split_by_1(y) << 1);
#endif
}

/**
 * @brief Converts a 2D float coordinate (x, y) into a 64-bit Morton Z-order key.
 * 
 * Maps IEEE-754 32-bit floats into monotonic unsigned integers via sign-bit manipulation,
 * then performs Morton interleaving.
 * 
 *
 * @param x Real (X) coordinate.
 *
 * @param y Imaginary (Y) coordinate.
 * @return @c uint64_t 64-bit Morton key.
 */
inline uint64_t float_to_morton(float x, float y) noexcept {
    auto float_to_ordered_uint = [](float f) -> uint32_t {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        uint32_t mask = (static_cast<int32_t>(u) >> 31) | 0x80000000u;
        return u ^ mask;
    };
    return morton_interleave(float_to_ordered_uint(x), float_to_ordered_uint(y));
}

/**
 * @struct MortonCompare
 * @brief Comparator for complex numbers using 2D Morton Z-order curve keys.
 */
struct MortonCompare {
    using is_transparent = void;

    template <typename T, typename U>
    constexpr bool operator()(const std::complex<T>& a, const std::complex<U>& b) const noexcept {
        return float_to_morton(static_cast<float>(a.real()), static_cast<float>(a.imag())) < 
               float_to_morton(static_cast<float>(b.real()), static_cast<float>(b.imag()));
    }
};

/**
 * @brief Sorts a contiguous span of complex numbers along the 2D Morton Z-order curve.
 * 
 * Uses a 4-pass 16-bit Radix Sort across 64-bit keys for large arrays (<tt>N >= 256</tt>),
 * achieving <tt>O(N)</tt> time complexity and preserving 2D spatial locality.
 * 
 * @tparam T Floating-point or arithmetic component type of the complex numbers.
 *
 * @param data Contiguous span of complex numbers to sort in-place.
 */
template <typename T>
void sort_complex_morton(std::span<std::complex<T>> data) {
    if (data.size() < 256) {
        std::sort(data.begin(), data.end(), MortonCompare{});
        return;
    }

    struct Element {
        uint64_t key;
        std::complex<T> val;
    };

    std::vector<Element> src(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        src[i].val = data[i];
        src[i].key = float_to_morton(static_cast<float>(data[i].real()), static_cast<float>(data[i].imag()));
    }

    std::vector<Element> dst(data.size());
    Element* src_ptr = src.data();
    Element* dst_ptr = dst.data();
    size_t n = data.size();

    // 4 passes of 16-bit radix sort (64 bits total)
    for (int shift = 0; shift < 64; shift += 16) {
        std::size_t count[65536] = {0};
        for (size_t i = 0; i < n; ++i) {
            count[(src_ptr[i].key >> shift) & 0xFFFF]++;
        }

        std::size_t total = 0;
        for (int i = 0; i < 65536; ++i) {
            std::size_t oldCount = count[i];
            count[i] = total;
            total += oldCount;
        }

        for (size_t i = 0; i < n; ++i) {
            std::size_t bucket = (src_ptr[i].key >> shift) & 0xFFFF;
            dst_ptr[count[bucket]++] = src_ptr[i];
        }

        std::swap(src_ptr, dst_ptr);
    }

    // 4 passes (even number), so src_ptr points to the originally allocated `src` vector buffer
    for (size_t i = 0; i < n; ++i) {
        data[i] = src_ptr[i].val;
    }
}

} // namespace algoat::numerics
