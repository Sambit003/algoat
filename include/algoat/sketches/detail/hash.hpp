#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#if defined(__GNUC__) || defined(__clang__)
#define ALGOAT_HASH_LIKELY(x) (__builtin_expect(!!(x), 1))
#define ALGOAT_HASH_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define ALGOAT_HASH_LIKELY(x) (x)
#define ALGOAT_HASH_UNLIKELY(x) (x)
#endif

namespace algoat::sketches::detail {

/// SplitMix64 fast 64-bit avalanche mixer (Guy Steele, Doug Lea, Christine Flood, 2014)
[[nodiscard]] constexpr uint64_t splitmix64(uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/// 128-bit multiplication helper for 64-bit hashing
[[nodiscard]] inline uint64_t wymum_mix128(uint64_t a, uint64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(r ^ (r >> 64));
#else
    uint64_t ha = a >> 32, la = static_cast<uint32_t>(a);
    uint64_t hb = b >> 32, lb = static_cast<uint32_t>(b);
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32);
    uint64_t c = t < rl;
    uint64_t lo = t + (rm1 << 32);
    c += lo < t;
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    return lo ^ hi;
#endif
}

[[nodiscard]] inline uint64_t read_u64_le(const uint8_t* p) noexcept {
    uint64_t v = 0;
    std::memcpy(&v, p, 8);
    return v;
}

[[nodiscard]] inline uint64_t read_u32_le(const uint8_t* p) noexcept {
    uint32_t v = 0;
    std::memcpy(&v, p, 4);
    return v;
}

[[nodiscard]] inline uint64_t read_u24_le(const uint8_t* p, size_t k) noexcept {
    return (static_cast<uint64_t>(p[0]) << 16) | (static_cast<uint64_t>(p[k >> 1]) << 8) | p[k - 1];
}

/// Portable Wyhash v3 core for 64-bit hashing of arbitrary byte buffers
[[nodiscard]] inline uint64_t wyhash(const void* key, size_t len, uint64_t seed = 0) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(key);
    static constexpr uint64_t WYHASH_CONSTANTS[4] = {0x2d358dccaa6c78a5ULL, 0x8bb84b93962eacc9ULL,
                                                     0x4b33a62ed433d4a3ULL, 0x4d5a2da51de1aa47ULL};
    uint64_t a = 0;
    uint64_t b = 0;
    seed ^= wymum_mix128(seed ^ WYHASH_CONSTANTS[0], WYHASH_CONSTANTS[1]);
    if (ALGOAT_HASH_LIKELY(len <= 16)) {
        if (ALGOAT_HASH_LIKELY(len >= 4)) {
            a = (read_u32_le(p) << 32) | read_u32_le(p + ((len >> 3) << 2));
            b = (read_u32_le(p + len - 4) << 32) | read_u32_le(p + len - 4 - ((len >> 3) << 2));
        } else if (ALGOAT_HASH_LIKELY(len > 0)) {
            a = read_u24_le(p, len);
            b = 0;
        } else {
            a = 0;
            b = 0;
        }
    } else {
        size_t i = len;
        if (ALGOAT_HASH_UNLIKELY(i > 48)) {
            uint64_t seed1 = seed;
            uint64_t seed2 = seed;
            do {
                seed =
                    wymum_mix128(read_u64_le(p) ^ WYHASH_CONSTANTS[1], read_u64_le(p + 8) ^ seed);
                seed1 = wymum_mix128(read_u64_le(p + 16) ^ WYHASH_CONSTANTS[2],
                                     read_u64_le(p + 24) ^ seed1);
                seed2 = wymum_mix128(read_u64_le(p + 32) ^ WYHASH_CONSTANTS[3],
                                     read_u64_le(p + 40) ^ seed2);
                p += 48;
                i -= 48;
            } while (ALGOAT_HASH_UNLIKELY(i > 48));
            seed ^= seed1 ^ seed2;
        }
        while (ALGOAT_HASH_UNLIKELY(i > 16)) {
            seed = wymum_mix128(read_u64_le(p) ^ WYHASH_CONSTANTS[1], read_u64_le(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = read_u64_le(p + i - 16);
        b = read_u64_le(p + i - 8);
    }
    a ^= WYHASH_CONSTANTS[1];
    b ^= seed;
    return wymum_mix128(a ^ WYHASH_CONSTANTS[0], b ^ len);
}

/// Generic item hasher mapping any C++ type to a well-distributed 64-bit integer
template <typename T> [[nodiscard]] inline uint64_t hash_item(const T& item) noexcept {
    using Decayed = std::decay_t<T>;
    if constexpr (std::is_integral_v<Decayed> || std::is_enum_v<Decayed>) {
        return splitmix64(static_cast<uint64_t>(item));
    } else if constexpr (std::is_floating_point_v<Decayed>) {
        uint64_t bits = 0;
        if constexpr (sizeof(Decayed) <= 4) {
            uint32_t b32 = 0;
            std::memcpy(&b32, &item, sizeof(item));
            bits = b32;
        } else {
            std::memcpy(&bits, &item, sizeof(item));
        }
        return splitmix64(bits);
    } else if constexpr (std::is_pointer_v<Decayed>) {
        return splitmix64(reinterpret_cast<uintptr_t>(item));
    } else if constexpr (std::is_same_v<Decayed, std::string> ||
                         std::is_same_v<Decayed, std::string_view>) {
        return wyhash(item.data(), item.size());
    } else if constexpr (requires {
                             { item.data() } -> std::contiguous_iterator;
                             { item.size() } -> std::convertible_to<std::size_t>;
                         }) {
        return wyhash(item.data(), item.size() * sizeof(typename Decayed::value_type));
    } else {
        return splitmix64(static_cast<uint64_t>(std::hash<Decayed>{}(item)));
    }
}

} // namespace algoat::sketches::detail

#undef ALGOAT_HASH_LIKELY
#undef ALGOAT_HASH_UNLIKELY
