#pragma once

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__AVX2__)
#include <immintrin.h>
#define ALGOAT_HASH_AVX2 1
#endif
#endif

namespace algoat::hash {

// ─── Forward Declarations & Concepts ──────────────────────────────────────────

using HighwayKey = std::array<uint64_t, 4>; ///< 256-bit secret key

/// \brief Concept for types that can be viewed as a contiguous byte span
template <typename T>
concept ByteSpannable = requires(const T& t) {
    { std::as_bytes(std::span{t}) } -> std::convertible_to<std::span<const std::byte>>;
};

// ─── Internal Implementation Details ──────────────────────────────────────────

namespace detail {

// Endian-safe helpers
[[nodiscard]] inline constexpr uint64_t swap64(uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(x);
#elif defined(_MSC_VER)
    return _byteswap_uint64(x);
#else
    return (((x >> 56) & 0xffULL) | ((x >> 40) & 0xff00ULL) | ((x >> 24) & 0xff0000ULL) |
            ((x >> 8) & 0xff000000ULL) | ((x << 8) & 0xff00000000ULL) |
            ((x << 24) & 0xff0000000000ULL) | ((x << 40) & 0xff000000000000ULL) |
            ((x << 56) & 0xff00000000000000ULL));
#endif
}

[[nodiscard]] inline constexpr uint32_t swap32(uint32_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#elif defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return (((x >> 24) & 0xffU) | ((x >> 8) & 0xff00U) | ((x << 8) & 0xff0000U) |
            ((x << 24) & 0xff000000U));
#endif
}

[[nodiscard]] inline uint64_t read_le64(const void* p) noexcept {
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    if constexpr (std::endian::native == std::endian::big) {
        v = swap64(v);
    }
    return v;
}

[[nodiscard]] inline uint32_t read_le32(const void* p) noexcept {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    if constexpr (std::endian::native == std::endian::big) {
        v = swap32(v);
    }
    return v;
}

// ─── xxHash3 Implementation ───────────────────────────────────────────────────

namespace xxh3 {

inline constexpr uint32_t PRIME32_1 = 0x9E3779B1U;
inline constexpr uint32_t PRIME32_2 = 0x85EBCA77U;
inline constexpr uint32_t PRIME32_3 = 0xC2B2AE3DU;
inline constexpr uint32_t PRIME32_4 = 0x27D4EB2FU;
inline constexpr uint32_t PRIME32_5 = 0x165667B1U;

inline constexpr uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
inline constexpr uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
inline constexpr uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
inline constexpr uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
inline constexpr uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;

inline constexpr uint64_t PRIME_MX1 = 0x165667919E3779F9ULL;
inline constexpr uint64_t PRIME_MX2 = 0x9FB21C651E98DF25ULL;

alignas(64) inline constexpr uint8_t kDefaultSecret[192] = {
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
};

struct Pair128 {
    uint64_t low64;
    uint64_t high64;
};

[[nodiscard]] inline Pair128 mult64to128(uint64_t lhs, uint64_t rhs) noexcept {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = static_cast<__uint128_t>(lhs) * static_cast<__uint128_t>(rhs);
    return Pair128{static_cast<uint64_t>(p), static_cast<uint64_t>(p >> 64)};
#else
    uint64_t ha = lhs >> 32, la = static_cast<uint32_t>(lhs);
    uint64_t hb = rhs >> 32, lb = static_cast<uint32_t>(rhs);
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32);
    uint64_t c = (t < rl);
    uint64_t lo = t + (rm1 << 32);
    c += (lo < t);
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    return Pair128{lo, hi};
#endif
}

[[nodiscard]] inline uint64_t mul128_fold64(uint64_t lhs, uint64_t rhs) noexcept {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = static_cast<__uint128_t>(lhs) * static_cast<__uint128_t>(rhs);
    return static_cast<uint64_t>(p) ^ static_cast<uint64_t>(p >> 64);
#else
    Pair128 p = mult64to128(lhs, rhs);
    return p.low64 ^ p.high64;
#endif
}

[[nodiscard]] inline uint64_t avalanche(uint64_t h64) noexcept {
    h64 ^= (h64 >> 37);
    h64 *= PRIME_MX1;
    h64 ^= (h64 >> 32);
    return h64;
}

[[nodiscard]] inline uint64_t avalanche_xxh64(uint64_t h64) noexcept {
    h64 ^= (h64 >> 33);
    h64 *= PRIME64_2;
    h64 ^= (h64 >> 29);
    h64 *= PRIME64_3;
    h64 ^= (h64 >> 32);
    return h64;
}

[[nodiscard]] inline uint64_t rrmxmx(uint64_t h64, uint64_t len) noexcept {
    h64 ^= std::rotl(h64, 49) ^ std::rotl(h64, 24);
    h64 *= PRIME_MX2;
    h64 ^= (h64 >> 35) + len;
    h64 *= PRIME_MX2;
    return h64 ^ (h64 >> 28);
}

[[nodiscard]] inline uint64_t mix16B(const uint8_t* in, const uint8_t* sec,
                                     uint64_t seed) noexcept {
    uint64_t in_lo = read_le64(in);
    uint64_t in_hi = read_le64(in + 8);
    return mul128_fold64(in_lo ^ (read_le64(sec) + seed), in_hi ^ (read_le64(sec + 8) - seed));
}

[[nodiscard]] inline Pair128 mix32B(Pair128 acc, const uint8_t* in1, const uint8_t* in2,
                                    const uint8_t* sec, uint64_t seed) noexcept {
    acc.low64 += mix16B(in1, sec + 0, seed);
    acc.low64 ^= read_le64(in2) + read_le64(in2 + 8);
    acc.high64 += mix16B(in2, sec + 16, seed);
    acc.high64 ^= read_le64(in1) + read_le64(in1 + 8);
    return acc;
}

inline void derive_secret_from_seed(uint8_t* dest, uint64_t seed) noexcept {
    for (std::size_t i = 0; i < 12; ++i) {
        uint64_t lo = read_le64(kDefaultSecret + 16 * i) + seed;
        uint64_t hi = read_le64(kDefaultSecret + 16 * i + 8) - seed;
        std::memcpy(dest + 16 * i, &lo, sizeof(lo));
        std::memcpy(dest + 16 * i + 8, &hi, sizeof(hi));
    }
}

// ─── xxHash3 Bulk AVX2 / Scalar Loops ─────────────────────────────────────────

inline void accumulate_512_scalar(uint64_t* acc, const uint8_t* in, const uint8_t* sec) noexcept {
    for (std::size_t lane = 0; lane < 8; ++lane) {
        uint64_t data_val = read_le64(in + lane * 8);
        uint64_t data_key = data_val ^ read_le64(sec + lane * 8);
        acc[lane ^ 1] += data_val;
        acc[lane] += static_cast<uint64_t>(static_cast<uint32_t>(data_key)) * (data_key >> 32);
    }
}

inline void scramble_acc_scalar(uint64_t* acc, const uint8_t* sec) noexcept {
    for (std::size_t lane = 0; lane < 8; ++lane) {
        uint64_t key64 = read_le64(sec + lane * 8);
        uint64_t acc64 = acc[lane];
        acc64 ^= (acc64 >> 47);
        acc64 ^= key64;
        acc64 *= PRIME32_1;
        acc[lane] = acc64;
    }
}

#if defined(ALGOAT_HASH_AVX2)
inline void accumulate_512_avx2(uint64_t* acc, const uint8_t* in, const uint8_t* sec) noexcept {
    const auto* xin = reinterpret_cast<const __m256i*>(in);
    const auto* xsec = reinterpret_cast<const __m256i*>(sec);

    for (std::size_t i = 0; i < 2; ++i) {
        __m256i acc_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i * 4));
        __m256i data_vec = _mm256_loadu_si256(xin + i);
        __m256i key_vec = _mm256_loadu_si256(xsec + i);
        __m256i data_key = _mm256_xor_si256(data_vec, key_vec);
        __m256i data_key_lo = _mm256_srli_epi64(data_key, 32);
        __m256i product = _mm256_mul_epu32(data_key, data_key_lo);
        __m256i data_swap = _mm256_shuffle_epi32(data_vec, _MM_SHUFFLE(1, 0, 3, 2));
        __m256i sum = _mm256_add_epi64(acc_vec, data_swap);
        __m256i res = _mm256_add_epi64(product, sum);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i * 4), res);
    }
}

inline void scramble_acc_avx2(uint64_t* acc, const uint8_t* sec) noexcept {
    const auto* xsec = reinterpret_cast<const __m256i*>(sec);
    const __m256i prime32 = _mm256_set1_epi32(static_cast<int>(PRIME32_1));

    for (std::size_t i = 0; i < 2; ++i) {
        __m256i acc_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i * 4));
        __m256i shifted = _mm256_srli_epi64(acc_vec, 47);
        __m256i data_vec = _mm256_xor_si256(acc_vec, shifted);
        __m256i key_vec = _mm256_loadu_si256(xsec + i);
        __m256i data_key = _mm256_xor_si256(data_vec, key_vec);

        __m256i data_key_hi = _mm256_srli_epi64(data_key, 32);
        __m256i prod_lo = _mm256_mul_epu32(data_key, prime32);
        __m256i prod_hi = _mm256_mul_epu32(data_key_hi, prime32);
        __m256i res = _mm256_add_epi64(prod_lo, _mm256_slli_epi64(prod_hi, 32));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i * 4), res);
    }
}
#endif

inline void accumulate_512(uint64_t* acc, const uint8_t* in, const uint8_t* sec) noexcept {
#if defined(ALGOAT_HASH_AVX2)
    accumulate_512_avx2(acc, in, sec);
#else
    accumulate_512_scalar(acc, in, sec);
#endif
}

inline void scramble_acc(uint64_t* acc, const uint8_t* sec) noexcept {
#if defined(ALGOAT_HASH_AVX2)
    scramble_acc_avx2(acc, sec);
#else
    scramble_acc_scalar(acc, sec);
#endif
}

[[nodiscard]] inline uint64_t mix_2accs(const uint64_t* acc, const uint8_t* sec) noexcept {
    return mul128_fold64(acc[0] ^ read_le64(sec), acc[1] ^ read_le64(sec + 8));
}

[[nodiscard]] inline uint64_t merge_accs(const uint64_t* acc, const uint8_t* sec,
                                         uint64_t start) noexcept {
    uint64_t result = start;
    for (std::size_t i = 0; i < 4; ++i) {
        result += mix_2accs(acc + 2 * i, sec + 16 * i);
    }
    return avalanche(result);
}

// ─── xxHash3 Public Kernels ───────────────────────────────────────────────────

inline void accumulate_blocks(uint64_t acc[8], const uint8_t* in, std::size_t len,
                              const uint8_t* active_sec) noexcept {
    constexpr std::size_t block_len = 1024;
    std::size_t nb_blocks = (len - 1) / block_len;
    for (std::size_t n = 0; n < nb_blocks; ++n) {
        const uint8_t* block = in + n * block_len;
        for (std::size_t s = 0; s < 16; ++s) {
            accumulate_512(acc, block + s * 64, active_sec + s * 8);
        }
        scramble_acc(acc, active_sec + 192 - 64);
    }

    std::size_t rem = (len - 1) - (block_len * nb_blocks);
    std::size_t nb_stripes = rem / 64;
    const uint8_t* last_block = in + nb_blocks * block_len;
    for (std::size_t s = 0; s < nb_stripes; ++s) {
        accumulate_512(acc, last_block + s * 64, active_sec + s * 8);
    }
    accumulate_512(acc, in + len - 64, active_sec + 192 - 64 - 7);
}

[[nodiscard]] inline uint64_t hash64(const uint8_t* in, std::size_t len, uint64_t seed,
                                     const uint8_t* sec = kDefaultSecret) noexcept {
    if (len == 0) {
        uint64_t bitflip = read_le64(sec + 56) ^ read_le64(sec + 64);
        return avalanche_xxh64(seed ^ bitflip);
    }
    if (len <= 3) {
        uint8_t c1 = in[0];
        uint8_t c2 = in[len >> 1];
        uint8_t c3 = in[len - 1];
        uint32_t combined = (static_cast<uint32_t>(c1) << 16) | (static_cast<uint32_t>(c2) << 24) |
                            (static_cast<uint32_t>(c3) << 0) | (static_cast<uint32_t>(len) << 8);
        uint64_t bitflip = (read_le32(sec) ^ read_le32(sec + 4)) + seed;
        uint64_t keyed = static_cast<uint64_t>(combined) ^ bitflip;
        return avalanche_xxh64(keyed);
    }
    if (len <= 8) {
        uint64_t modified_seed =
            seed ^ (static_cast<uint64_t>(swap32(static_cast<uint32_t>(seed))) << 32);
        uint32_t input1 = read_le32(in);
        uint32_t input2 = read_le32(in + len - 4);
        uint64_t bitflip = (read_le64(sec + 8) ^ read_le64(sec + 16)) - modified_seed;
        uint64_t input64 = input2 + (static_cast<uint64_t>(input1) << 32);
        uint64_t keyed = input64 ^ bitflip;
        return rrmxmx(keyed, len);
    }
    if (len <= 16) {
        uint64_t bitflip1 = (read_le64(sec + 24) ^ read_le64(sec + 32)) + seed;
        uint64_t bitflip2 = (read_le64(sec + 40) ^ read_le64(sec + 48)) - seed;
        uint64_t in_lo = read_le64(in) ^ bitflip1;
        uint64_t in_hi = read_le64(in + len - 8) ^ bitflip2;
        uint64_t acc = len + swap64(in_lo) + in_hi + mul128_fold64(in_lo, in_hi);
        return avalanche(acc);
    }
    if (len <= 128) {
        uint64_t acc = len * PRIME64_1;
        if (len > 32) {
            if (len > 64) {
                if (len > 96) {
                    acc += mix16B(in + 48, sec + 96, seed);
                    acc += mix16B(in + len - 64, sec + 112, seed);
                }
                acc += mix16B(in + 32, sec + 64, seed);
                acc += mix16B(in + len - 48, sec + 80, seed);
            }
            acc += mix16B(in + 16, sec + 32, seed);
            acc += mix16B(in + len - 32, sec + 48, seed);
        }
        acc += mix16B(in + 0, sec + 0, seed);
        acc += mix16B(in + len - 16, sec + 16, seed);
        return avalanche(acc);
    }
    if (len <= 240) {
        uint64_t acc = len * PRIME64_1;
        for (std::size_t i = 0; i < 8; ++i) {
            acc += mix16B(in + 16 * i, sec + 16 * i, seed);
        }
        acc = avalanche(acc);
        uint64_t acc_end = mix16B(in + len - 16, sec + 119, seed);
        std::size_t nb_rounds = len / 16;
        for (std::size_t i = 8; i < nb_rounds; ++i) {
            acc_end += mix16B(in + 16 * i, sec + 16 * (i - 8) + 3, seed);
        }
        return avalanche(acc + acc_end);
    }

    // Long inputs (> 240 bytes)
    alignas(64) uint8_t custom_sec[192];
    const uint8_t* active_sec = sec;
    if (seed != 0) {
        derive_secret_from_seed(custom_sec, seed);
        active_sec = custom_sec;
    }

    alignas(64) uint64_t acc[8] = {PRIME32_3, PRIME64_1, PRIME64_2, PRIME64_3,
                                   PRIME64_4, PRIME32_2, PRIME64_5, PRIME32_1};
    accumulate_blocks(acc, in, len, active_sec);
    return merge_accs(acc, active_sec + 11, len * PRIME64_1);
}

[[nodiscard]] inline Pair128 hash128(const uint8_t* in, std::size_t len, uint64_t seed,
                                     const uint8_t* sec = kDefaultSecret) noexcept {
    if (len == 0) {
        uint64_t bitflipl = read_le64(sec + 64) ^ read_le64(sec + 72);
        uint64_t bitfliph = read_le64(sec + 80) ^ read_le64(sec + 88);
        return Pair128{avalanche_xxh64(seed ^ bitflipl), avalanche_xxh64(seed ^ bitfliph)};
    }
    if (len <= 3) {
        uint8_t c1 = in[0];
        uint8_t c2 = in[len >> 1];
        uint8_t c3 = in[len - 1];
        uint32_t combinedl = (static_cast<uint32_t>(c1) << 16) | (static_cast<uint32_t>(c2) << 24) |
                             (static_cast<uint32_t>(c3) << 0) | (static_cast<uint32_t>(len) << 8);
        uint32_t combinedh = std::rotl(swap32(combinedl), 13);
        uint64_t bitflipl = (read_le32(sec) ^ read_le32(sec + 4)) + seed;
        uint64_t bitfliph = (read_le32(sec + 8) ^ read_le32(sec + 12)) - seed;
        return Pair128{avalanche_xxh64(static_cast<uint64_t>(combinedl) ^ bitflipl),
                       avalanche_xxh64(static_cast<uint64_t>(combinedh) ^ bitfliph)};
    }
    if (len <= 8) {
        uint64_t modified_seed =
            seed ^ (static_cast<uint64_t>(swap32(static_cast<uint32_t>(seed))) << 32);
        uint32_t in_lo = read_le32(in);
        uint32_t in_hi = read_le32(in + len - 4);
        uint64_t in_64 = in_lo + (static_cast<uint64_t>(in_hi) << 32);
        uint64_t bitflip = (read_le64(sec + 16) ^ read_le64(sec + 24)) + modified_seed;
        uint64_t keyed = in_64 ^ bitflip;

        Pair128 m128 = mult64to128(keyed, PRIME64_1 + (len << 2));
        m128.high64 += (m128.low64 << 1);
        m128.low64 ^= (m128.high64 >> 3);

        m128.low64 = (m128.low64 ^ (m128.low64 >> 35)) * PRIME_MX2;
        m128.low64 = m128.low64 ^ (m128.low64 >> 28);
        m128.high64 = avalanche(m128.high64);
        return m128;
    }
    if (len <= 16) {
        uint64_t bitflipl = (read_le64(sec + 32) ^ read_le64(sec + 40)) - seed;
        uint64_t bitfliph = (read_le64(sec + 48) ^ read_le64(sec + 56)) + seed;
        uint64_t in_lo = read_le64(in);
        uint64_t in_hi = read_le64(in + len - 8);
        Pair128 m128 = mult64to128(in_lo ^ in_hi ^ bitflipl, PRIME64_1);
        m128.low64 += static_cast<uint64_t>(len - 1) << 54;
        in_hi ^= bitfliph;
        m128.high64 +=
            in_hi + static_cast<uint64_t>(static_cast<uint32_t>(in_hi)) * (PRIME32_2 - 1);
        m128.low64 ^= swap64(m128.high64);

        Pair128 h128 = mult64to128(m128.low64, PRIME64_2);
        h128.high64 += m128.high64 * PRIME64_2;
        h128.low64 = avalanche(h128.low64);
        h128.high64 = avalanche(h128.high64);
        return h128;
    }
    if (len <= 128) {
        Pair128 acc = {len * PRIME64_1, 0};
        if (len > 32) {
            if (len > 64) {
                if (len > 96) {
                    acc = mix32B(acc, in + 48, in + len - 64, sec + 96, seed);
                }
                acc = mix32B(acc, in + 32, in + len - 48, sec + 64, seed);
            }
            acc = mix32B(acc, in + 16, in + len - 32, sec + 32, seed);
        }
        acc = mix32B(acc, in, in + len - 16, sec, seed);

        uint64_t low = acc.low64 + acc.high64;
        uint64_t high =
            (acc.low64 * PRIME64_1) + (acc.high64 * PRIME64_4) + ((len - seed) * PRIME64_2);
        return Pair128{avalanche(low), static_cast<uint64_t>(0) - avalanche(high)};
    }
    if (len <= 240) {
        Pair128 acc = {len * PRIME64_1, 0};
        for (std::size_t i = 32; i < 160; i += 32) {
            acc = mix32B(acc, in + i - 32, in + i - 16, sec + i - 32, seed);
        }
        acc.low64 = avalanche(acc.low64);
        acc.high64 = avalanche(acc.high64);
        for (std::size_t i = 160; i <= len; i += 32) {
            acc = mix32B(acc, in + i - 32, in + i - 16, sec + 3 + i - 160, seed);
        }
        acc = mix32B(acc, in + len - 16, in + len - 32, sec + 103, static_cast<uint64_t>(0) - seed);

        uint64_t low = acc.low64 + acc.high64;
        uint64_t high =
            (acc.low64 * PRIME64_1) + (acc.high64 * PRIME64_4) + ((len - seed) * PRIME64_2);
        return Pair128{avalanche(low), static_cast<uint64_t>(0) - avalanche(high)};
    }

    // Long inputs (> 240 bytes)
    alignas(64) uint8_t custom_sec[192];
    const uint8_t* active_sec = sec;
    if (seed != 0) {
        derive_secret_from_seed(custom_sec, seed);
        active_sec = custom_sec;
    }

    alignas(64) uint64_t acc[8] = {PRIME32_3, PRIME64_1, PRIME64_2, PRIME64_3,
                                   PRIME64_4, PRIME32_2, PRIME64_5, PRIME32_1};
    accumulate_blocks(acc, in, len, active_sec);

    uint64_t low64 = merge_accs(acc, active_sec + 11, len * PRIME64_1);
    uint64_t high64 = merge_accs(acc, active_sec + 192 - 64 - 11, ~(len * PRIME64_2));
    return Pair128{low64, high64};
}

} // namespace xxh3

// ─── WyHash Implementation (Canonical v4) ─────────────────────────────────────

namespace wyhash_impl {

inline constexpr std::array<uint64_t, 4> kDefaultSecret = {
    0x2d358dccaa6c78a5ULL, 0x8bb84b93962eacc9ULL, 0x4b33a62ed433d4a3ULL, 0x4d5a2da51de1aa47ULL};

inline void wymum(uint64_t* a, uint64_t* b) noexcept {
#if defined(__SIZEOF_INT128__)
    __uint128_t r = static_cast<__uint128_t>(*a) * static_cast<__uint128_t>(*b);
    *a = static_cast<uint64_t>(r);
    *b = static_cast<uint64_t>(r >> 64);
#else
    uint64_t ha = *a >> 32, la = static_cast<uint32_t>(*a);
    uint64_t hb = *b >> 32, lb = static_cast<uint32_t>(*b);
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32);
    uint64_t c = (t < rl);
    uint64_t lo = t + (rm1 << 32);
    c += (lo < t);
    uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
    *a = lo;
    *b = hi;
#endif
}

[[nodiscard]] inline uint64_t wymix(uint64_t a, uint64_t b) noexcept {
    wymum(&a, &b);
    return a ^ b;
}

[[nodiscard]] inline uint64_t read3(const uint8_t* p, std::size_t k) noexcept {
    return (static_cast<uint64_t>(p[0]) << 16) | (static_cast<uint64_t>(p[k >> 1]) << 8) |
           static_cast<uint64_t>(p[k - 1]);
}

[[nodiscard]] inline uint64_t hash(const uint8_t* p, std::size_t len, uint64_t seed,
                                   const uint64_t* secret = kDefaultSecret.data()) noexcept {
    seed ^= wymix(seed ^ secret[0], secret[1]);
    uint64_t a = 0;
    uint64_t b = 0;

    if (len <= 16) {
        if (len >= 4) {
            a = (static_cast<uint64_t>(read_le32(p)) << 32) | read_le32(p + ((len >> 3) << 2));
            b = (static_cast<uint64_t>(read_le32(p + len - 4)) << 32) |
                read_le32(p + len - 4 - ((len >> 3) << 2));
        } else if (len > 0) {
            a = read3(p, len);
            b = 0;
        } else {
            a = 0;
            b = 0;
        }
    } else {
        std::size_t i = len;
        if (i >= 48) {
            uint64_t see1 = seed;
            uint64_t see2 = seed;
            do {
                seed = wymix(read_le64(p) ^ secret[1], read_le64(p + 8) ^ seed);
                see1 = wymix(read_le64(p + 16) ^ secret[2], read_le64(p + 24) ^ see1);
                see2 = wymix(read_le64(p + 32) ^ secret[3], read_le64(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while (i >= 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = wymix(read_le64(p) ^ secret[1], read_le64(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = read_le64(p + i - 16);
        b = read_le64(p + i - 8);
    }

    a ^= secret[1];
    b ^= seed;
    wymum(&a, &b);
    return wymix(a ^ secret[0] ^ len, b ^ secret[1]);
}

} // namespace wyhash_impl

// ─── HighwayHash Implementation ───────────────────────────────────────────────

namespace highway {

inline constexpr std::array<uint64_t, 4> kInit0 = {0xdbe6d5d5fe4cce2fULL, 0xa4093822299f31d0ULL,
                                                   0x13198a2e03707344ULL, 0x243f6a8885a308d3ULL};

inline constexpr std::array<uint64_t, 4> kInit1 = {0x3bd39e10cb0ef593ULL, 0xc0acf169b5f18a8cULL,
                                                   0xbe5466cf34e90c6cULL, 0x452821e638d01377ULL};

[[nodiscard]] inline constexpr uint64_t rotate64_by32(uint64_t x) noexcept {
    return (x >> 32) | (x << 32);
}

[[nodiscard]] inline constexpr uint64_t rotate32_lane(uint64_t v, int count) noexcept {
    uint32_t lo = std::rotl(static_cast<uint32_t>(v), count);
    uint32_t hi = std::rotl(static_cast<uint32_t>(v >> 32), count);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

[[nodiscard]] inline constexpr uint64_t zipper_merge0(uint64_t v1, uint64_t v0) noexcept {
    return ((((v0 & 0xff000000ULL) | (v1 & 0xff00000000ULL)) >> 24) |
            (((v0 & 0xff0000000000ULL) | (v1 & 0xff000000000000ULL)) >> 16) | (v0 & 0xff0000ULL) |
            ((v0 & 0xff00ULL) << 32) | ((v1 & 0xff00000000000000ULL) >> 8) | (v0 << 56));
}

[[nodiscard]] inline constexpr uint64_t zipper_merge1(uint64_t v1, uint64_t v0) noexcept {
    return ((((v1 & 0xff000000ULL) | (v0 & 0xff00000000ULL)) >> 24) | (v1 & 0xff0000ULL) |
            ((v1 & 0xff0000000000ULL) >> 16) | ((v1 & 0xff00ULL) << 24) |
            ((v0 & 0xff000000000000ULL) >> 8) | ((v1 & 0xffULL) << 48) |
            (v0 & 0xff00000000000000ULL));
}

inline void modular_reduction(uint64_t a3_unmasked, uint64_t a2, uint64_t a1, uint64_t a0,
                              uint64_t* h1, uint64_t* h0) noexcept {
    const uint64_t a3 = a3_unmasked & 0x3FFFFFFFFFFFFFFFULL;
    *h1 = a1 ^ ((a3 << 1) | (a2 >> 63)) ^ ((a3 << 2) | (a2 >> 62));
    *h0 = a0 ^ (a2 << 1) ^ (a2 << 2);
}

struct State {
    std::array<uint64_t, 4> v0;
    std::array<uint64_t, 4> v1;
    std::array<uint64_t, 4> mul0;
    std::array<uint64_t, 4> mul1;

    explicit State(const HighwayKey& key) noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            mul0[i] = kInit0[i];
            mul1[i] = kInit1[i];
            v0[i] = kInit0[i] ^ key[i];
            v1[i] = kInit1[i] ^ rotate64_by32(key[i]);
        }
    }

    void update_scalar(const uint64_t packet[4]) noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            v1[i] += packet[i];
            v1[i] += mul0[i];
        }
        for (std::size_t i = 0; i < 4; ++i) {
            const uint32_t v1_32 = static_cast<uint32_t>(v1[i]);
            mul0[i] ^= static_cast<uint64_t>(v1_32) * (v0[i] >> 32);

            v0[i] += mul1[i];

            const uint32_t v0_32 = static_cast<uint32_t>(v0[i]);
            mul1[i] ^= static_cast<uint64_t>(v0_32) * (v1[i] >> 32);
        }

        v0[0] += zipper_merge0(v1[1], v1[0]);
        v0[1] += zipper_merge1(v1[1], v1[0]);
        v0[2] += zipper_merge0(v1[3], v1[2]);
        v0[3] += zipper_merge1(v1[3], v1[2]);

        v1[0] += zipper_merge0(v0[1], v0[0]);
        v1[1] += zipper_merge1(v0[1], v0[0]);
        v1[2] += zipper_merge0(v0[3], v0[2]);
        v1[3] += zipper_merge1(v0[3], v0[2]);
    }

#if defined(ALGOAT_HASH_AVX2)
    void update_avx2(const uint64_t packet[4]) noexcept {
        const __m256i shuffle_mask =
            _mm256_setr_epi8(0x03, 0x0c, 0x02, 0x05, 0x0e, 0x01, 0x0f, 0x00, 0x0b, 0x04, 0x0a, 0x0d,
                             0x09, 0x06, 0x08, 0x07, 0x03, 0x0c, 0x02, 0x05, 0x0e, 0x01, 0x0f, 0x00,
                             0x0b, 0x04, 0x0a, 0x0d, 0x09, 0x06, 0x08, 0x07);

        __m256i mv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v0.data()));
        __m256i mv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v1.data()));
        __m256i mmul0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mul0.data()));
        __m256i mmul1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mul1.data()));
        __m256i mpacket = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packet));

        mv1 = _mm256_add_epi64(mv1, mpacket);
        mv1 = _mm256_add_epi64(mv1, mmul0);

        __m256i v0_hi = _mm256_srli_epi64(mv0, 32);
        __m256i prod0 = _mm256_mul_epu32(mv1, v0_hi);
        mmul0 = _mm256_xor_si256(mmul0, prod0);

        mv0 = _mm256_add_epi64(mv0, mmul1);

        __m256i v1_hi = _mm256_srli_epi64(mv1, 32);
        __m256i prod1 = _mm256_mul_epu32(mv0, v1_hi);
        mmul1 = _mm256_xor_si256(mmul1, prod1);

        __m256i zm_v1 = _mm256_shuffle_epi8(mv1, shuffle_mask);
        mv0 = _mm256_add_epi64(mv0, zm_v1);

        __m256i zm_v0 = _mm256_shuffle_epi8(mv0, shuffle_mask);
        mv1 = _mm256_add_epi64(mv1, zm_v0);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v0.data()), mv0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v1.data()), mv1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mul0.data()), mmul0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mul1.data()), mmul1);
    }
#endif

    void update(const uint64_t packet[4]) noexcept {
#if defined(ALGOAT_HASH_AVX2)
        update_avx2(packet);
#else
        update_scalar(packet);
#endif
    }

    void permute_and_update() noexcept {
        uint64_t permuted[4] = {rotate64_by32(v0[2]), rotate64_by32(v0[3]), rotate64_by32(v0[0]),
                                rotate64_by32(v0[1])};
        update(permuted);
    }
};

inline void process_data(State& state, const uint8_t* bytes, std::size_t size) noexcept {
    const std::size_t size_mod32 = size % 32;
    const std::size_t num_packets = size / 32;

    for (std::size_t p = 0; p < num_packets; ++p) {
        uint64_t packet[4];
        std::memcpy(packet, bytes + p * 32, 32);
        if constexpr (std::endian::native == std::endian::big) {
            for (auto& w : packet)
                w = swap64(w);
        }
        state.update(packet);
    }

    if (size_mod32 > 0) {
        const uint64_t mod32_pair = (static_cast<uint64_t>(size_mod32) << 32) + size_mod32;
        for (std::size_t i = 0; i < 4; ++i) {
            state.v0[i] += mod32_pair;
            state.v1[i] = rotate32_lane(state.v1[i], static_cast<int>(size_mod32));
        }

        uint8_t packet[32] = {0};
        const std::size_t remainder_prefix = size_mod32 & ~3ULL;
        const std::size_t size_mod4 = size_mod32 & 3ULL;

        if (remainder_prefix > 0) {
            std::memcpy(packet, bytes + size - size_mod32, remainder_prefix);
        }

        if ((size_mod32 & 16) != 0) {
            for (std::size_t i = 0; i < 4; ++i) {
                packet[28 + i] = bytes[size - 4 + i];
            }
        } else if (size_mod4 != 0) {
            const uint8_t* rem = bytes + (size - size_mod4);
            packet[16 + 0] = rem[0];
            packet[16 + 1] = rem[size_mod4 >> 1];
            packet[16 + 2] = rem[size_mod4 - 1];
        }

        uint64_t packet64[4];
        std::memcpy(packet64, packet, 32);
        if constexpr (std::endian::native == std::endian::big) {
            for (auto& w : packet64)
                w = swap64(w);
        }
        state.update(packet64);
    }
}

[[nodiscard]] inline uint64_t hash64(const uint8_t* bytes, std::size_t size,
                                     const HighwayKey& key) noexcept {
    State state(key);
    process_data(state, bytes, size);

    for (int i = 0; i < 4; ++i) {
        state.permute_and_update();
    }

    return state.v0[0] + state.v1[0] + state.mul0[0] + state.mul1[0];
}

[[nodiscard]] inline std::array<uint64_t, 2> hash128(const uint8_t* bytes, std::size_t size,
                                                     const HighwayKey& key) noexcept {
    State state(key);
    process_data(state, bytes, size);

    for (int i = 0; i < 6; ++i) {
        state.permute_and_update();
    }

    return {state.v0[0] + state.mul0[0] + state.v1[2] + state.mul1[2],
            state.v0[1] + state.mul0[1] + state.v1[3] + state.mul1[3]};
}

[[nodiscard]] inline std::array<uint64_t, 4> hash256(const uint8_t* bytes, std::size_t size,
                                                     const HighwayKey& key) noexcept {
    State state(key);
    process_data(state, bytes, size);

    for (int i = 0; i < 10; ++i) {
        state.permute_and_update();
    }

    std::array<uint64_t, 4> result{};
    modular_reduction(state.v1[1] + state.mul1[1], state.v1[0] + state.mul1[0],
                      state.v0[1] + state.mul0[1], state.v0[0] + state.mul0[0], &result[1],
                      &result[0]);

    modular_reduction(state.v1[3] + state.mul1[3], state.v1[2] + state.mul1[2],
                      state.v0[3] + state.mul0[3], state.v0[2] + state.mul0[2], &result[3],
                      &result[2]);
    return result;
}

} // namespace highway

} // namespace detail

// ─── Public API ───────────────────────────────────────────────────────────────

// ─── xxHash3 ──────────────────────────────────────────────────────────────────

/// \brief xxHash3 64-bit hash of arbitrary byte data.
[[nodiscard]] inline uint64_t xxhash3_64(std::span<const std::byte> data,
                                         uint64_t seed = 0) noexcept {
    return detail::xxh3::hash64(reinterpret_cast<const uint8_t*>(data.data()), data.size(), seed);
}

/// \brief xxHash3 64-bit hash of string view data.
[[nodiscard]] inline uint64_t xxhash3_64(std::string_view str, uint64_t seed = 0) noexcept {
    return detail::xxh3::hash64(reinterpret_cast<const uint8_t*>(str.data()), str.size(), seed);
}

/// \brief xxHash3 128-bit hash as pair (low64, high64).
[[nodiscard]] inline std::array<uint64_t, 2> xxhash3_128(std::span<const std::byte> data,
                                                         uint64_t seed = 0) noexcept {
    auto p =
        detail::xxh3::hash128(reinterpret_cast<const uint8_t*>(data.data()), data.size(), seed);
    return {p.low64, p.high64};
}

/// \brief xxHash3 128-bit hash of string view data.
[[nodiscard]] inline std::array<uint64_t, 2> xxhash3_128(std::string_view str,
                                                         uint64_t seed = 0) noexcept {
    auto p = detail::xxh3::hash128(reinterpret_cast<const uint8_t*>(str.data()), str.size(), seed);
    return {p.low64, p.high64};
}

// ─── WyHash ───────────────────────────────────────────────────────────────────

/// \brief WyHash 64-bit hash, optimized for small keys.
[[nodiscard]] inline uint64_t wyhash(std::span<const std::byte> data, uint64_t seed = 0) noexcept {
    return detail::wyhash_impl::hash(reinterpret_cast<const uint8_t*>(data.data()), data.size(),
                                     seed);
}

/// \brief WyHash 64-bit hash of string view data.
[[nodiscard]] inline uint64_t wyhash(std::string_view str, uint64_t seed = 0) noexcept {
    return detail::wyhash_impl::hash(reinterpret_cast<const uint8_t*>(str.data()), str.size(),
                                     seed);
}

// ─── HighwayHash ──────────────────────────────────────────────────────────────

/// \brief HighwayHash 64-bit with user-supplied key (DoS-resistant).
[[nodiscard]] inline uint64_t highway_hash_64(std::span<const std::byte> data,
                                              const HighwayKey& key) noexcept {
    return detail::highway::hash64(reinterpret_cast<const uint8_t*>(data.data()), data.size(), key);
}

/// \brief HighwayHash 64-bit of string view data.
[[nodiscard]] inline uint64_t highway_hash_64(std::string_view str,
                                              const HighwayKey& key) noexcept {
    return detail::highway::hash64(reinterpret_cast<const uint8_t*>(str.data()), str.size(), key);
}

/// \brief HighwayHash 128-bit with user-supplied key (DoS-resistant) as (h0, h1).
[[nodiscard]] inline std::array<uint64_t, 2> highway_hash_128(std::span<const std::byte> data,
                                                              const HighwayKey& key) noexcept {
    return detail::highway::hash128(reinterpret_cast<const uint8_t*>(data.data()), data.size(),
                                    key);
}

/// \brief HighwayHash 128-bit of string view data.
[[nodiscard]] inline std::array<uint64_t, 2> highway_hash_128(std::string_view str,
                                                              const HighwayKey& key) noexcept {
    return highway_hash_128(std::as_bytes(std::span{str.data(), str.size()}), key);
}

/// \brief HighwayHash 256-bit as 4-element uint64 array.
[[nodiscard]] inline std::array<uint64_t, 4> highway_hash_256(std::span<const std::byte> data,
                                                              const HighwayKey& key) noexcept {
    return detail::highway::hash256(reinterpret_cast<const uint8_t*>(data.data()), data.size(),
                                    key);
}

/// \brief HighwayHash 256-bit of string view data.
[[nodiscard]] inline std::array<uint64_t, 4> highway_hash_256(std::string_view str,
                                                              const HighwayKey& key) noexcept {
    return detail::highway::hash256(reinterpret_cast<const uint8_t*>(str.data()), str.size(), key);
}

// ─── std::hash-compatible CPO Adaptors ────────────────────────────────────────

namespace detail {

template <typename HasherFunc, typename T>
[[nodiscard]] inline std::size_t dispatch_hasher(const T& val, HasherFunc hasher) noexcept {
    if constexpr (ByteSpannable<T>) {
        return static_cast<std::size_t>(hasher(std::as_bytes(std::span{val})));
    } else if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_pointer_v<T>) {
        return static_cast<std::size_t>(hasher(std::as_bytes(std::span<const T, 1>{&val, 1})));
    } else {
        return std::hash<T>{}(val);
    }
}

} // namespace detail

/// \brief Hasher CPO using xxHash3_64 for use with std::unordered_map / unordered_set.
template <typename T> struct XxHash3Hasher {
    [[nodiscard]] std::size_t operator()(const T& val) const noexcept {
        return detail::dispatch_hasher(
            val, [](std::span<const std::byte> sp) noexcept { return xxhash3_64(sp); });
    }
};

/// \brief Hasher CPO using WyHash for use with std::unordered_map / unordered_set.
template <typename T> struct WyHasher {
    [[nodiscard]] std::size_t operator()(const T& val) const noexcept {
        return detail::dispatch_hasher(
            val, [](std::span<const std::byte> sp) noexcept { return wyhash(sp); });
    }
};

} // namespace algoat::hash
