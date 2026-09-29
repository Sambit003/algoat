/**
 * @file hybrid_sfc.hpp
 * @brief Two-Tier Hybrid Morton-Hilbert Spatial Sorting Engine.
 *
 * Implements a Pareto-optimal hybrid space-filling curve that combines the
 * near-zero compute overhead of Morton (Z-order) sorting with the continuous
 * cache locality of the Hilbert curve.
 *
 * @par Hybrid Architecture:
 * - Top-Level Partitioning: Upper 16 bits of X and Y are interleaved using Morton
 *   Z-order to form a 32-bit coarse tile key.
 * - Intra-Tile Micro-Sorting: Lower 16 bits of X and Y are mapped using a Hilbert
 *   curve to form a 32-bit fine key.
 *
 * @par Sorting Strategy:
 * - For <tt>N < 256</tt>: @c std::sort with a transparent comparator.
 * - For <tt>N >= 256</tt>: 4-pass 16-bit Radix Sort across the 64-bit hybrid keys.
 */

#pragma once

#include "algoat/numerics/morton.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace algoat::numerics {

/**
 * @brief Converts 16-bit 2D coordinates into a 32-bit Hilbert index.
 */
inline uint32_t hilbert2D_16(uint16_t x, uint16_t y) noexcept {
    uint32_t d = 0;
    for (uint32_t s = 32768; s > 0; s >>= 1) {
        uint32_t rx = (x & s) > 0;
        uint32_t ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            uint16_t t = x;
            x = y;
            y = t;
        }
    }
    return d;
}

/**
 * @brief Converts a 2D float coordinate (x, y) into a 64-bit Hybrid Morton-Hilbert key.
 */
inline uint64_t float_to_hybrid_sfc(float x, float y) noexcept {
    auto float_to_ordered_uint = [](float f) -> uint32_t {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        uint32_t mask = (static_cast<int32_t>(u) >> 31) | 0x80000000u;
        return u ^ mask;
    };

    uint32_t qx = float_to_ordered_uint(x);
    uint32_t qy = float_to_ordered_uint(y);

    uint16_t x_hi = qx >> 16;
    uint16_t y_hi = qy >> 16;
    uint16_t x_lo = qx & 0xFFFF;
    uint16_t y_lo = qy & 0xFFFF;

    // Top 32 bits: Morton interleaving of the high 16 bits
    uint64_t coarse_morton = morton_interleave(x_hi, y_hi);

    // Bottom 32 bits: Hilbert mapping of the low 16 bits
    uint64_t fine_hilbert = hilbert2D_16(x_lo, y_lo);

    return (coarse_morton << 32) | fine_hilbert;
}

/**
 * @struct HybridCompare
 * @brief Comparator for complex numbers using 64-bit Hybrid Morton-Hilbert keys.
 */
struct HybridCompare {
    using is_transparent = void;

    template <typename T, typename U>
    constexpr bool operator()(const std::complex<T>& a, const std::complex<U>& b) const noexcept {
        return float_to_hybrid_sfc(static_cast<float>(a.real()), static_cast<float>(a.imag())) <
               float_to_hybrid_sfc(static_cast<float>(b.real()), static_cast<float>(b.imag()));
    }
};

/**
 * @brief Sorts a contiguous span of complex numbers along the Hybrid Morton-Hilbert curve.
 */
template <typename T> void sort_complex_hybrid(std::span<std::complex<T>> data) {
    if (data.size() < 256) {
        std::sort(data.begin(), data.end(), HybridCompare{});
        return;
    }

    struct Element {
        uint64_t key;
        std::complex<T> val;
    };

    std::vector<Element> src(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        src[i].val = data[i];
        src[i].key = float_to_hybrid_sfc(static_cast<float>(data[i].real()),
                                         static_cast<float>(data[i].imag()));
    }

    std::vector<Element> dst(data.size());
    Element* src_ptr = src.data();
    Element* dst_ptr = dst.data();
    size_t n = data.size();

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

    for (size_t i = 0; i < n; ++i) {
        data[i] = src_ptr[i].val;
    }
}

} // namespace algoat::numerics
