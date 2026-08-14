#pragma once

#include <complex>
#include <tuple>
#include <cstdint>
#include <bit>
#include <cstring>

#if defined(__BMI2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace algoat::numerics {

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

inline uint64_t float_to_morton(float x, float y) noexcept {
    auto float_to_ordered_uint = [](float f) -> uint32_t {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        uint32_t mask = (static_cast<int32_t>(u) >> 31) | 0x80000000u;
        return u ^ mask;
    };
    return morton_interleave(float_to_ordered_uint(x), float_to_ordered_uint(y));
}

struct MortonCompare {
    using is_transparent = void;

    template <typename T, typename U>
    constexpr bool operator()(const std::complex<T>& a, const std::complex<U>& b) const noexcept {
        return float_to_morton(static_cast<float>(a.real()), static_cast<float>(a.imag())) < 
               float_to_morton(static_cast<float>(b.real()), static_cast<float>(b.imag()));
    }
};

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

    // 4 passes of 16-bit radix sort
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

    // 4 passes (even), so src_ptr points to the originally allocated `src` vector buffer
    for (size_t i = 0; i < n; ++i) {
        data[i] = src_ptr[i].val;
    }
}

} // namespace algoat::numerics
