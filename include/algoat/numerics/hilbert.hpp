#pragma once

#include "algoat/sorting/introsort.hpp"

#include <bit>
#include <compare>
#include <complex>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace algoat::numerics {

// Float to monotonic unsigned integer using strictly branchless bit-flipping
template <std::floating_point F> inline auto float_to_ordered_int(F f) noexcept {
    if constexpr (sizeof(F) == 4) {
        uint32_t u = std::bit_cast<uint32_t>(f);
        return u ^ ((static_cast<int32_t>(u) >> 31) | 0x80000000);
    } else {
        uint64_t u = std::bit_cast<uint64_t>(f);
        return u ^ ((static_cast<int64_t>(u) >> 63) | 0x8000000000000000ULL);
    }
}

// Branchless 2D coordinate-to-Hilbert key encoder
[[nodiscard]] inline uint64_t float_to_hilbert2d(float x, float y) noexcept {
    uint32_t ix = float_to_ordered_int(x);
    uint32_t iy = float_to_ordered_int(y);
    uint64_t d = 0;

    for (int s = 31; s >= 0; --s) {
        uint32_t rx = (ix >> s) & 1;
        uint32_t ry = (iy >> s) & 1;

        d = (d << 2) | ((3 * rx) ^ ry);

        uint32_t swap_mask = -((rx == 0) & 1);
        uint32_t flip_mask = -((ry == 1) & 1);
        uint32_t limit = (1U << s) - 1;

        uint32_t tx = ix ^ (flip_mask & swap_mask & limit);
        uint32_t ty = iy ^ (flip_mask & swap_mask & limit);

        ix = (ix & ~swap_mask) | (ty & swap_mask);
        iy = (iy & ~swap_mask) | (tx & swap_mask);
    }
    return d;
}

// 24-state SO(3) octahedral rotation lookup table (192 bytes)
constexpr uint8_t HILBERT_3D_TABLE[24][8] = {
    {8, 17, 27, 2, 15, 22, 28, 5},           {39, 46, 32, 41, 52, 13, 51, 10},
    {59, 48, 18, 65, 60, 55, 21, 70},        {49, 54, 26, 29, 72, 79, 83, 84},
    {92, 37, 103, 110, 91, 34, 96, 105},     {20, 95, 19, 88, 45, 118, 42, 113},
    {120, 127, 89, 94, 131, 132, 50, 53},    {58, 129, 61, 134, 107, 24, 108, 31},
    {66, 139, 145, 128, 69, 140, 150, 135},  {14, 77, 9, 74, 143, 124, 136, 123},
    {126, 119, 85, 156, 121, 112, 82, 155},  {67, 68, 160, 167, 90, 93, 169, 174},
    {171, 98, 172, 101, 0, 57, 7, 62},       {109, 86, 44, 175, 106, 81, 43, 168},
    {117, 180, 114, 179, 78, 71, 73, 64},    {183, 164, 38, 125, 176, 163, 33, 122},
    {161, 80, 166, 87, 130, 187, 133, 188},  {141, 138, 190, 185, 36, 35, 63, 56},
    {1, 146, 184, 75, 6, 149, 191, 76},      {47, 40, 4, 3, 182, 177, 157, 154},
    {152, 147, 159, 148, 97, 162, 102, 165}, {170, 173, 115, 116, 25, 30, 144, 151},
    {100, 99, 181, 178, 23, 16, 142, 137},   {158, 153, 111, 104, 189, 186, 12, 11},
};

// 3D coordinate-to-Hilbert key encoder using the 24-state SO(3) table
[[nodiscard]] inline uint64_t float_to_hilbert3d(float x, float y, float z) noexcept {
    uint32_t ix = float_to_ordered_int(x) >> 11;
    uint32_t iy = float_to_ordered_int(y) >> 11;
    uint32_t iz = float_to_ordered_int(z) >> 11;
    uint64_t d = 0;
    uint32_t state = 0;

    for (int s = 20; s >= 0; --s) {
        uint32_t xyz = (((ix >> s) & 1) << 2) | (((iy >> s) & 1) << 1) | ((iz >> s) & 1);
        uint8_t trans = HILBERT_3D_TABLE[state][xyz];
        d = (d << 3) | (trans & 7);
        state = trans >> 3;
    }
    return d;
}

template <typename T>
[[nodiscard]] inline uint64_t complex_to_hilbert2d(const std::complex<T>& c) noexcept {
    return float_to_hilbert2d(static_cast<float>(c.real()), static_cast<float>(c.imag()));
}

// Transparent comparator for STL compatibility
struct HilbertCompare {
    using is_transparent = void;
    template <typename T>
    bool operator()(const std::complex<T>& a, const std::complex<T>& b) const noexcept {
        return complex_to_hilbert2d(a) < complex_to_hilbert2d(b);
    }
};

// O(N) 4-pass 16-bit LSD Radix Sort along the Hilbert curve
template <typename T> void sort_complex_hilbert(std::span<std::complex<T>> data) {
    if (data.empty())
        return;

    struct Payload {
        uint64_t key;
        std::complex<T> val;
        auto operator<=>(const Payload& other) const {
            return key <=> other.key;
        }
        bool operator==(const Payload& other) const {
            return key == other.key;
        }
    };

    std::vector<Payload> buf(data.size() * 2);
    Payload* src = buf.data();
    Payload* dst = buf.data() + data.size();

    for (size_t i = 0; i < data.size(); ++i) {
        src[i].val = data[i];
        src[i].key = complex_to_hilbert2d(data[i]);
    }

    if (data.size() < 256) {
        algoat::sorting::IntroSort{}.sort(std::span<Payload>{src, data.size()});
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = src[i].val;
        }
        return;
    }

    auto get_bucket = [](uint64_t key, int shift) { return (key >> shift) & 0xFFFF; };

    for (int pass = 0; pass < 4; ++pass) {
        size_t count[65536] = {0};
        int shift = pass * 16;

        for (size_t i = 0; i < data.size(); ++i) {
            count[get_bucket(src[i].key, shift)]++;
        }

        size_t sum = 0;
        for (int i = 0; i < 65536; ++i) {
            size_t c = count[i];
            count[i] = sum;
            sum += c;
        }

        for (size_t i = 0; i < data.size(); ++i) {
            dst[count[get_bucket(src[i].key, shift)]++] = src[i];
        }

        std::swap(src, dst);
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = src[i].val;
    }
}

} // namespace algoat::numerics
