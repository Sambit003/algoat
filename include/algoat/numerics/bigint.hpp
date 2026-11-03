/**
 * @file bigint.hpp
 * @brief Arbitrary-Precision BigInt Engine with Karatsuba & Montgomery Multiplication.
 *
 * Implements high-performance arbitrary-precision integer arithmetic:
 * 1. Correctness across arbitrary bit-widths with C++20 standard two's complement / sign-magnitude
 * semantics.
 * 2. Small-Value Optimization (SVO): integers <= 256 bits stored inline without heap allocation.
 * 3. Schoolbook Multiplication accelerated with BMI2 intrinsics (_mulx_u64, _addcarry_u64).
 * 4. Subtractive Karatsuba Multiplication O(n^1.585) for medium and large operands without
 * carry-out limbs.
 * 5. Knuth Algorithm D multi-precision division with single-limb fast path.
 * 6. Montgomery Modular Multiplication (CIOS) and Exponentiation (mod_pow) for odd moduli.
 * 7. Extended Euclidean Algorithm for modular inverse.
 *
 * @author Algoat Development Team
 * @date 2026
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <climits>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace algoat::numerics {

namespace detail {

// Portable 64-bit add with carry
inline unsigned char add_carry_u64(unsigned char c_in, uint64_t a, uint64_t b,
                                   uint64_t* out) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_IX86) && !defined(__i386__)
    return _addcarry_u64(c_in, a, b, reinterpret_cast<unsigned long long*>(out));
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + c_in;
    *out = static_cast<uint64_t>(sum);
    return static_cast<unsigned char>(sum >> 64);
#else
    uint64_t s = a + b + c_in;
    *out = s;
    return (s < a || (c_in && s == a)) ? 1 : 0;
#endif
}

// Portable 64-bit subtract with borrow
inline unsigned char sub_borrow_u64(unsigned char b_in, uint64_t a, uint64_t b,
                                    uint64_t* out) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_IX86) && !defined(__i386__)
    return _subborrow_u64(b_in, a, b, reinterpret_cast<unsigned long long*>(out));
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 diff = static_cast<unsigned __int128>(a) - b - b_in;
    *out = static_cast<uint64_t>(diff);
    return static_cast<unsigned char>((diff >> 127) & 1);
#else
    uint64_t d = a - b - b_in;
    *out = d;
    return (a < b || (b_in && a == b)) ? 1 : 0;
#endif
}

// Portable 64x64 -> 128 unsigned multiply
inline uint64_t mulx_u64(uint64_t a, uint64_t b, uint64_t* hi) noexcept {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64)) && !defined(_M_IX86) &&          \
    !defined(__i386__)
    unsigned long long hi_val = 0;
    unsigned long long lo = _mulx_u64(a, b, &hi_val);
    *hi = static_cast<uint64_t>(hi_val);
    return static_cast<uint64_t>(lo);
#elif defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
    return _umul128(a, b, hi);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
    *hi = static_cast<uint64_t>(prod >> 64);
    return static_cast<uint64_t>(prod);
#else
    uint64_t a_lo = a & 0xffffffffULL, a_hi = a >> 32;
    uint64_t b_lo = b & 0xffffffffULL, b_hi = b >> 32;
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    uint64_t mid = p1 + (p0 >> 32) + (p2 & 0xffffffffULL);
    *hi = p3 + (p2 >> 32) + (mid >> 32);
    return (p0 & 0xffffffffULL) | (mid << 32);
#endif
}

// Portable 64x64 -> 128 unsigned multiply-accumulate with carry: *acc += a * b + carry_in
inline uint64_t mul_add_carry(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t* acc) noexcept {
#if defined(__BMI2__) && defined(__SIZEOF_INT128__)
    unsigned long long hi_val = 0;
    unsigned long long lo = _mulx_u64(a, b, &hi_val);
    unsigned __int128 sum = static_cast<unsigned __int128>(lo) + *acc + carry_in;
    *acc = static_cast<uint64_t>(sum);
    return static_cast<uint64_t>(hi_val) + static_cast<uint64_t>(sum >> 64);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
    unsigned __int128 sum = prod + *acc + carry_in;
    *acc = static_cast<uint64_t>(sum);
    return static_cast<uint64_t>(sum >> 64);
#elif defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
    uint64_t hi = 0;
    uint64_t lo = _umul128(a, b, &hi);
    unsigned char c1 = _addcarry_u64(0, lo, *acc, acc);
    uint64_t hi_carry = 0;
    _addcarry_u64(c1, hi, carry_in, &hi_carry);
    return hi_carry;
#else
    uint64_t hi = 0;
    uint64_t lo = mulx_u64(a, b, &hi);
    unsigned char c1 = add_carry_u64(0, lo, *acc, acc);
    uint64_t hi_carry = 0;
    add_carry_u64(c1, hi, carry_in, &hi_carry);
    return hi_carry;
#endif
}

// Portable 128 / 64 -> 64 division with remainder (hi must be < divisor)
inline uint64_t div128_64(uint64_t hi, uint64_t lo, uint64_t divisor, uint64_t* rem) noexcept {
#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
    return _udiv128(hi, lo, divisor, rem);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 dividend = (static_cast<unsigned __int128>(hi) << 64) | lo;
    *rem = static_cast<uint64_t>(dividend % divisor);
    return static_cast<uint64_t>(dividend / divisor);
#else
    uint64_t q = 0;
    uint64_t r = hi;
    for (int i = 63; i >= 0; --i) {
        r = (r << 1) | ((lo >> i) & 1ULL);
        if (r >= divisor) {
            r -= divisor;
            q |= (1ULL << i);
        }
    }
    *rem = r;
    return q;
#endif
}

// Raw schoolbook multiplication on flat arrays accelerated with BMI2 / wide arithmetic
inline void multiply_schoolbook_raw(const uint64_t* a, size_t n, const uint64_t* b, size_t m,
                                    uint64_t* r) noexcept {
    std::memset(r, 0, (n + m) * sizeof(uint64_t));
    for (size_t i = 0; i < n; ++i) {
        uint64_t ai = a[i];
        if (ai == 0)
            continue;
        uint64_t carry = 0;
        for (size_t j = 0; j < m; ++j) {
            carry = mul_add_carry(ai, b[j], carry, &r[i + j]);
        }
        r[i + m] = carry;
    }
}

// Compute |x - y| into out (padded to out_len with 0), returns sgn(x - y)
inline int sub_abs_raw(const uint64_t* x, size_t x_len, const uint64_t* y, size_t y_len,
                       uint64_t* out, size_t out_len) noexcept {
    // Compare x and y from MSB down to 0
    size_t max_len = std::max(x_len, y_len);
    int cmp = 0;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(max_len) - 1; i >= 0; --i) {
        uint64_t xi = (static_cast<size_t>(i) < x_len) ? x[i] : 0;
        uint64_t yi = (static_cast<size_t>(i) < y_len) ? y[i] : 0;
        if (xi > yi) {
            cmp = 1;
            break;
        }
        if (xi < yi) {
            cmp = -1;
            break;
        }
    }

    if (cmp == 0) {
        std::memset(out, 0, out_len * sizeof(uint64_t));
        return 0;
    }

    const uint64_t* big = (cmp > 0) ? x : y;
    size_t big_len = (cmp > 0) ? x_len : y_len;
    const uint64_t* small = (cmp > 0) ? y : x;
    size_t small_len = (cmp > 0) ? y_len : x_len;

    unsigned char borrow = 0;
    for (size_t i = 0; i < big_len; ++i) {
        uint64_t bi = big[i];
        uint64_t si = (i < small_len) ? small[i] : 0;
        borrow = sub_borrow_u64(borrow, bi, si, &out[i]);
    }
    for (size_t i = big_len; i < out_len; ++i) {
        out[i] = 0;
    }
    return cmp;
}

// Fast inlined 4x4 schoolbook multiplication
inline void mul_4x4_inline(const uint64_t* a, const uint64_t* b, uint64_t* r) noexcept {
    std::memset(r, 0, 8 * sizeof(uint64_t));
    for (size_t i = 0; i < 4; ++i) {
        uint64_t ai = a[i];
        if (ai == 0)
            continue;
        uint64_t carry = 0;
        for (size_t j = 0; j < 4; ++j) {
            carry = mul_add_carry(ai, b[j], carry, &r[i + j]);
        }
        r[i + 4] = carry;
    }
}

// Dedicated unrolled 8x8 (512-bit) Karatsuba multiplication kernel
inline void karatsuba_8x8(const uint64_t* a, const uint64_t* b, uint64_t* r) noexcept {
    uint64_t z0[8];
    uint64_t z2[8];
    mul_4x4_inline(a, b, z0);
    mul_4x4_inline(a + 4, b + 4, z2);

    uint64_t delta_a[4];
    uint64_t delta_b[4];
    int sign_a = sub_abs_raw(a, 4, a + 4, 4, delta_a, 4);
    int sign_b = sub_abs_raw(b, 4, b + 4, 4, delta_b, 4);
    int sign_delta = sign_a * sign_b;

    uint64_t p_delta[8]{};
    if (sign_delta != 0) {
        mul_4x4_inline(delta_a, delta_b, p_delta);
    }

    uint64_t z1[9];
    std::memcpy(z1, z0, 8 * sizeof(uint64_t));
    z1[8] = 0;

    unsigned char carry = 0;
    for (size_t i = 0; i < 8; ++i) {
        carry = add_carry_u64(carry, z1[i], z2[i], &z1[i]);
    }
    z1[8] = carry;

    if (sign_delta == 1) {
        unsigned char borrow = 0;
        for (size_t i = 0; i < 8; ++i) {
            borrow = sub_borrow_u64(borrow, z1[i], p_delta[i], &z1[i]);
        }
        z1[8] -= borrow;
    } else if (sign_delta == -1) {
        carry = 0;
        for (size_t i = 0; i < 8; ++i) {
            carry = add_carry_u64(carry, z1[i], p_delta[i], &z1[i]);
        }
        z1[8] += carry;
    }

    std::memcpy(r, z0, 8 * sizeof(uint64_t));
    std::memcpy(r + 8, z2, 8 * sizeof(uint64_t));

    carry = 0;
    for (size_t i = 0; i < 9; ++i) {
        carry = add_carry_u64(carry, r[4 + i], z1[i], &r[4 + i]);
    }
    for (size_t i = 13; carry && i < 16; ++i) {
        carry = add_carry_u64(carry, r[i], 0, &r[i]);
    }
}

// Subtractive Karatsuba on flat limb arrays using caller-provided scratch
inline void karatsuba_raw(const uint64_t* a, const uint64_t* b, size_t n, uint64_t* r,
                          uint64_t* scratch) noexcept {
    if (n <= 4) {
        multiply_schoolbook_raw(a, n, b, n, r);
        return;
    }
    if (n == 8) {
        karatsuba_8x8(a, b, r);
        return;
    }

    size_t m = (n + 1) / 2;
    size_t high_len = n - m;

    const uint64_t* a0 = a;
    const uint64_t* a1 = a + m;
    const uint64_t* b0 = b;
    const uint64_t* b1 = b + m;

    uint64_t* delta_a = scratch;
    uint64_t* delta_b = scratch + m;
    uint64_t* p_delta = scratch + 2 * m;
    uint64_t* z1_buf = scratch + 4 * m;
    uint64_t* next_scratch = scratch + 7 * m;

    // 1. Z0 = a0 * b0 -> r[0 ... 2m-1]
    karatsuba_raw(a0, b0, m, r, next_scratch);

    // 2. Z2 = a1 * b1 -> r[2m ... 2n-1]
    if (high_len == m) {
        karatsuba_raw(a1, b1, m, r + 2 * m, next_scratch);
    } else {
        multiply_schoolbook_raw(a1, high_len, b1, high_len, r + 2 * m);
    }

    // 3. Delta_A = |a0 - a1|, Delta_B = |b0 - b1|
    int sign_a = sub_abs_raw(a0, m, a1, high_len, delta_a, m);
    int sign_b = sub_abs_raw(b0, m, b1, high_len, delta_b, m);
    int sign_delta = sign_a * sign_b;

    // 4. P_delta = Delta_A * Delta_B
    if (sign_delta != 0) {
        karatsuba_raw(delta_a, delta_b, m, p_delta, next_scratch);
    } else {
        std::memset(p_delta, 0, 2 * m * sizeof(uint64_t));
    }

    // 5. Compute Z1 = Z0 + Z2 - sign_delta * P_delta into z1_buf (size 2m + 1)
    std::memcpy(z1_buf, r, 2 * m * sizeof(uint64_t));
    z1_buf[2 * m] = 0;

    unsigned char carry = 0;
    for (size_t i = 0; i < 2 * high_len; ++i) {
        carry = add_carry_u64(carry, z1_buf[i], r[2 * m + i], &z1_buf[i]);
    }
    for (size_t i = 2 * high_len; i <= 2 * m; ++i) {
        carry = add_carry_u64(carry, z1_buf[i], 0, &z1_buf[i]);
    }

    if (sign_delta == 1) {
        unsigned char borrow = 0;
        for (size_t i = 0; i < 2 * m; ++i) {
            borrow = sub_borrow_u64(borrow, z1_buf[i], p_delta[i], &z1_buf[i]);
        }
        z1_buf[2 * m] -= borrow;
    } else if (sign_delta == -1) {
        carry = 0;
        for (size_t i = 0; i < 2 * m; ++i) {
            carry = add_carry_u64(carry, z1_buf[i], p_delta[i], &z1_buf[i]);
        }
        z1_buf[2 * m] += carry;
    }

    // 6. Accumulate z1_buf into r starting at offset m
    carry = 0;
    for (size_t i = 0; i <= 2 * m; ++i) {
        if (m + i < 2 * n) {
            carry = add_carry_u64(carry, r[m + i], z1_buf[i], &r[m + i]);
        }
    }
    for (size_t i = m + 2 * m + 1; carry && i < 2 * n; ++i) {
        carry = add_carry_u64(carry, r[i], 0, &r[i]);
    }
}

// Precompute -n0^(-1) mod 2^64 via Hensel's lemma / Newton-Raphson iteration
inline uint64_t compute_montgomery_n_prime(uint64_t n0) noexcept {
    assert((n0 & 1ULL) != 0ULL);
    uint64_t x = 1;
    for (int i = 0; i < 6; ++i) {
        x = x * (2ULL - n0 * x);
    }
    return 0ULL - x;
}

} // namespace detail

class BigInt {
public:
    // Small Value Optimization: inline storage for <= 256 bits (4 * 64-bit limbs)
    static constexpr size_t kSVOLimbs = 4;
    // Karatsuba threshold in 64-bit limbs (8 limbs = 512 bits)
    static constexpr size_t kKaratsubaThreshold = 8;

    // Constructors
    BigInt() noexcept : negative_(false), size_(0) {
        svo_limbs_[0] = svo_limbs_[1] = svo_limbs_[2] = svo_limbs_[3] = 0;
    }

    explicit BigInt(uint64_t val) noexcept : negative_(false), size_(val != 0 ? 1 : 0) {
        svo_limbs_[0] = val;
        svo_limbs_[1] = svo_limbs_[2] = svo_limbs_[3] = 0;
    }

    explicit BigInt(int64_t val) noexcept {
        svo_limbs_[1] = svo_limbs_[2] = svo_limbs_[3] = 0;
        if (val < 0) {
            negative_ = true;
            uint64_t uval = (val == INT64_MIN) ? static_cast<uint64_t>(INT64_MAX) + 1ULL
                                               : static_cast<uint64_t>(-val);
            svo_limbs_[0] = uval;
            size_ = 1;
        } else if (val > 0) {
            negative_ = false;
            svo_limbs_[0] = static_cast<uint64_t>(val);
            size_ = 1;
        } else {
            negative_ = false;
            svo_limbs_[0] = 0;
            size_ = 0;
        }
    }

    template <typename T,
              typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, int64_t> &&
                                          !std::is_same_v<T, uint64_t> && !std::is_same_v<T, bool>>>
    explicit BigInt(T val) noexcept {
        if constexpr (std::is_signed_v<T>) {
            *this = BigInt(static_cast<int64_t>(val));
        } else {
            *this = BigInt(static_cast<uint64_t>(val));
        }
    }

    explicit BigInt(std::string_view decimal_str) : BigInt() {
        if (decimal_str.empty()) {
            return;
        }
        size_t idx = 0;
        bool neg = false;
        if (decimal_str[0] == '-') {
            neg = true;
            idx = 1;
        } else if (decimal_str[0] == '+') {
            idx = 1;
        }

        if (idx >= decimal_str.size()) {
            throw std::invalid_argument("Invalid decimal string for BigInt");
        }

        // Process in 18-decimal-digit chunks (10^18 fits safely within uint64_t)
        constexpr size_t kChunkLen = 18;

        BigInt result;
        while (idx < decimal_str.size()) {
            size_t take = std::min(kChunkLen, decimal_str.size() - idx);
            uint64_t chunk_val = 0;
            uint64_t multiplier = 1;
            for (size_t i = 0; i < take; ++i) {
                char c = decimal_str[idx + i];
                if (c < '0' || c > '9') {
                    throw std::invalid_argument("Invalid character in decimal BigInt string");
                }
                chunk_val = chunk_val * 10 + static_cast<uint64_t>(c - '0');
                multiplier *= 10;
            }
            idx += take;

            result = result * BigInt(multiplier) + BigInt(chunk_val);
        }

        if (!result.is_zero() && neg) {
            result.negative_ = true;
        }
        *this = std::move(result);
    }

    explicit BigInt(std::string_view str, int base) : BigInt() {
        if (base < 2 || base > 36) {
            throw std::invalid_argument("Base must be between 2 and 36");
        }
        if (str.empty()) {
            return;
        }

        size_t idx = 0;
        bool neg = false;
        if (str[0] == '-') {
            neg = true;
            idx = 1;
        } else if (str[0] == '+') {
            idx = 1;
        }

        if (base == 16 && idx + 1 < str.size() && str[idx] == '0' &&
            (str[idx + 1] == 'x' || str[idx + 1] == 'X')) {
            idx += 2;
        } else if (base == 2 && idx + 1 < str.size() && str[idx] == '0' &&
                   (str[idx + 1] == 'b' || str[idx + 1] == 'B')) {
            idx += 2;
        } else if (base == 8 && idx + 1 < str.size() && str[idx] == '0' &&
                   (str[idx + 1] == 'o' || str[idx + 1] == 'O')) {
            idx += 2;
        }

        if (idx >= str.size()) {
            throw std::invalid_argument("Invalid string for BigInt");
        }

        // Strip leading zeros
        while (idx < str.size() && str[idx] == '0') {
            ++idx;
        }
        if (idx == str.size()) {
            *this = BigInt();
            return;
        }

        if (base == 16) {
            size_t hex_digits = str.size() - idx;
            size_t limbs_needed = (hex_digits + 15) / 16;
            allocate_exact(limbs_needed);

            auto l = limbs();
            for (size_t i = 0; i < limbs_needed; ++i) {
                l[i] = 0;
            }

            for (size_t i = 0; i < hex_digits; ++i) {
                char c = str[str.size() - 1 - i];
                uint64_t val = 0;
                if (c >= '0' && c <= '9') {
                    val = static_cast<uint64_t>(c - '0');
                } else if (c >= 'a' && c <= 'f') {
                    val = static_cast<uint64_t>(c - 'a' + 10);
                } else if (c >= 'A' && c <= 'F') {
                    val = static_cast<uint64_t>(c - 'A' + 10);
                } else {
                    throw std::invalid_argument("Invalid hex character in BigInt string");
                }

                size_t limb_idx = i / 16;
                size_t shift = (i % 16) * 4;
                l[limb_idx] |= (val << shift);
            }

            normalize();
            if (!is_zero() && neg) {
                negative_ = true;
            }
            return;
        }

        if (base == 10) {
            *this = BigInt(str.substr(neg ? 1 : 0));
            if (!is_zero() && neg) {
                negative_ = true;
            }
            return;
        }

        // Arbitrary base 2..36
        BigInt result;
        BigInt base_bi(base);
        for (; idx < str.size(); ++idx) {
            char c = str[idx];
            uint64_t val = 0;
            if (c >= '0' && c <= '9') {
                val = static_cast<uint64_t>(c - '0');
            } else if (c >= 'a' && c <= 'z') {
                val = static_cast<uint64_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'Z') {
                val = static_cast<uint64_t>(c - 'A' + 10);
            } else {
                throw std::invalid_argument("Invalid digit for specified base");
            }
            if (val >= static_cast<uint64_t>(base)) {
                throw std::invalid_argument("Digit out of range for specified base");
            }
            result = result * base_bi + BigInt(val);
        }

        if (!result.is_zero() && neg) {
            result.negative_ = true;
        }
        *this = std::move(result);
    }

    // Rule of Five
    BigInt(const BigInt& other) : negative_(other.negative_), size_(other.size_) {
        if (other.is_svo()) {
            std::memcpy(svo_limbs_, other.svo_limbs_, sizeof(svo_limbs_));
            heap_limbs_.clear();
        } else {
            heap_limbs_ = other.heap_limbs_;
        }
    }

    BigInt(BigInt&& other) noexcept : negative_(other.negative_), size_(other.size_) {
        if (other.is_svo()) {
            std::memcpy(svo_limbs_, other.svo_limbs_, sizeof(svo_limbs_));
        } else {
            heap_limbs_ = std::move(other.heap_limbs_);
        }
        other.size_ = 0;
        other.negative_ = false;
    }

    BigInt& operator=(const BigInt& other) {
        if (this != &other) {
            negative_ = other.negative_;
            size_ = other.size_;
            if (other.is_svo()) {
                std::memcpy(svo_limbs_, other.svo_limbs_, sizeof(svo_limbs_));
                heap_limbs_.clear();
                heap_limbs_.shrink_to_fit();
            } else {
                heap_limbs_ = other.heap_limbs_;
            }
        }
        return *this;
    }

    BigInt& operator=(BigInt&& other) noexcept {
        if (this != &other) {
            negative_ = other.negative_;
            size_ = other.size_;
            if (other.is_svo()) {
                std::memcpy(svo_limbs_, other.svo_limbs_, sizeof(svo_limbs_));
                heap_limbs_.clear();
                heap_limbs_.shrink_to_fit();
            } else {
                heap_limbs_ = std::move(other.heap_limbs_);
            }
            other.size_ = 0;
            other.negative_ = false;
        }
        return *this;
    }

    ~BigInt() = default;

    // Inspection
    [[nodiscard]] size_t num_bits() const noexcept {
        if (is_zero())
            return 0;
        auto l = limbs();
        uint64_t msb_limb = l[size_ - 1];
        size_t leading_zeros = static_cast<size_t>(std::countl_zero(msb_limb));
        return (size_ * 64) - leading_zeros;
    }

    [[nodiscard]] size_t num_limbs() const noexcept {
        return size_;
    }
    [[nodiscard]] bool is_zero() const noexcept {
        return size_ == 0;
    }
    [[nodiscard]] bool is_negative() const noexcept {
        return negative_ && !is_zero();
    }
    [[nodiscard]] int sign() const noexcept {
        if (is_zero())
            return 0;
        return negative_ ? -1 : 1;
    }

    [[nodiscard]] bool fits_int64() const noexcept {
        if (is_zero())
            return true;
        if (size_ > 1)
            return false;
        uint64_t val = limbs()[0];
        if (negative_) {
            return val <= (static_cast<uint64_t>(INT64_MAX) + 1ULL);
        }
        return val <= static_cast<uint64_t>(INT64_MAX);
    }

    [[nodiscard]] int64_t to_int64() const {
        if (!fits_int64()) {
            throw std::overflow_error("BigInt does not fit in int64_t");
        }
        if (is_zero())
            return 0;
        uint64_t val = limbs()[0];
        if (negative_) {
            if (val == static_cast<uint64_t>(INT64_MAX) + 1ULL) {
                return INT64_MIN;
            }
            return -static_cast<int64_t>(val);
        }
        return static_cast<int64_t>(val);
    }

    // Comparison operators
    [[nodiscard]] std::strong_ordering compare(const BigInt& rhs) const noexcept {
        int s1 = sign();
        int s2 = rhs.sign();
        if (s1 != s2) {
            return s1 <=> s2;
        }
        if (s1 == 0) {
            return std::strong_ordering::equal;
        }

        std::strong_ordering mag_cmp = compare_magnitudes(limbs(), rhs.limbs());
        if (s1 < 0) {
            if (mag_cmp == std::strong_ordering::less)
                return std::strong_ordering::greater;
            if (mag_cmp == std::strong_ordering::greater)
                return std::strong_ordering::less;
            return std::strong_ordering::equal;
        }
        return mag_cmp;
    }

    [[nodiscard]] bool operator==(const BigInt& rhs) const noexcept {
        if (size_ != rhs.size_ || negative_ != rhs.negative_) {
            return false;
        }
        if (size_ == 0) {
            return true;
        }
        auto l1 = limbs();
        auto l2 = rhs.limbs();
        return std::equal(l1.begin(), l1.end(), l2.begin());
    }

    [[nodiscard]] bool operator!=(const BigInt& rhs) const noexcept {
        return !(*this == rhs);
    }
    [[nodiscard]] bool operator<(const BigInt& rhs) const noexcept {
        return compare(rhs) == std::strong_ordering::less;
    }
    [[nodiscard]] bool operator<=(const BigInt& rhs) const noexcept {
        auto ord = compare(rhs);
        return ord == std::strong_ordering::less || ord == std::strong_ordering::equal;
    }
    [[nodiscard]] bool operator>(const BigInt& rhs) const noexcept {
        return compare(rhs) == std::strong_ordering::greater;
    }
    [[nodiscard]] bool operator>=(const BigInt& rhs) const noexcept {
        auto ord = compare(rhs);
        return ord == std::strong_ordering::greater || ord == std::strong_ordering::equal;
    }

#if !defined(__GNUC__) || defined(__clang__) || (__GNUC__ >= 15)
    [[nodiscard]] std::strong_ordering operator<=>(const BigInt& rhs) const noexcept {
        return compare(rhs);
    }
#endif

    // Unary negation
    [[nodiscard]] BigInt operator-() const {
        if (is_zero()) {
            return *this;
        }
        BigInt res = *this;
        res.negative_ = !res.negative_;
        return res;
    }

    // Arithmetic operators
    [[nodiscard]] BigInt operator+(const BigInt& rhs) const {
        if (is_zero())
            return rhs;
        if (rhs.is_zero())
            return *this;

        if (negative_ == rhs.negative_) {
            BigInt res = add_magnitudes(limbs(), rhs.limbs());
            res.negative_ = negative_;
            return res;
        }

        std::strong_ordering cmp = compare_magnitudes(limbs(), rhs.limbs());
        if (cmp == std::strong_ordering::equal) {
            return BigInt();
        }
        if (cmp == std::strong_ordering::greater) {
            BigInt res = sub_magnitudes(limbs(), rhs.limbs());
            res.negative_ = negative_;
            return res;
        } else {
            BigInt res = sub_magnitudes(rhs.limbs(), limbs());
            res.negative_ = rhs.negative_;
            return res;
        }
    }

    [[nodiscard]] BigInt operator-(const BigInt& rhs) const {
        return *this + (-rhs);
    }

    BigInt& operator+=(const BigInt& rhs) {
        *this = *this + rhs;
        return *this;
    }

    BigInt& operator-=(const BigInt& rhs) {
        *this = *this - rhs;
        return *this;
    }

    [[nodiscard]] BigInt operator*(const BigInt& rhs) const {
        if (is_zero() || rhs.is_zero()) {
            return BigInt();
        }

        BigInt res;
        size_t max_limbs = std::max(size_, rhs.size_);
        if (max_limbs <= kKaratsubaThreshold) {
            res = multiply_schoolbook(limbs(), rhs.limbs());
        } else {
            res = multiply_karatsuba(limbs(), rhs.limbs());
        }

        res.negative_ = (negative_ != rhs.negative_);
        res.normalize();
        return res;
    }

    BigInt& operator*=(const BigInt& rhs) {
        *this = *this * rhs;
        return *this;
    }

    [[nodiscard]] BigInt operator/(const BigInt& rhs) const {
        BigInt q, r;
        divide_knuth(*this, rhs, q, r);
        return q;
    }

    [[nodiscard]] BigInt operator%(const BigInt& rhs) const {
        BigInt q, r;
        divide_knuth(*this, rhs, q, r);
        return r;
    }

    BigInt& operator/=(const BigInt& rhs) {
        *this = *this / rhs;
        return *this;
    }

    BigInt& operator%=(const BigInt& rhs) {
        *this = *this % rhs;
        return *this;
    }

    // Bitwise operators
    [[nodiscard]] BigInt operator<<(size_t bits) const {
        if (is_zero() || bits == 0) {
            return *this;
        }
        size_t limb_shift = bits / 64;
        size_t bit_shift = bits % 64;

        BigInt res;
        res.allocate_exact(size_ + limb_shift + 1);
        auto rl = res.limbs();
        for (size_t i = 0; i < limb_shift; ++i) {
            rl[i] = 0;
        }

        auto src = limbs();
        if (bit_shift == 0) {
            for (size_t i = 0; i < size_; ++i) {
                rl[i + limb_shift] = src[i];
            }
            rl[size_ + limb_shift] = 0;
        } else {
            uint64_t carry = 0;
            for (size_t i = 0; i < size_; ++i) {
                rl[i + limb_shift] = (src[i] << bit_shift) | carry;
                carry = src[i] >> (64 - bit_shift);
            }
            rl[size_ + limb_shift] = carry;
        }

        res.negative_ = negative_;
        res.normalize();
        return res;
    }

    // Two's complement bitwise logic
    [[nodiscard]] BigInt operator~() const {
        return -(*this + BigInt(1LL));
    }

    [[nodiscard]] BigInt operator>>(size_t bits) const {
        if (is_zero() || bits == 0) {
            return *this;
        }
        if (is_negative()) {
            return ~((~*this) >> bits);
        }

        size_t limb_shift = bits / 64;
        size_t bit_shift = bits % 64;

        if (limb_shift >= size_) {
            return BigInt();
        }

        size_t new_len = size_ - limb_shift;
        BigInt res;
        res.allocate_exact(new_len);
        auto rl = res.limbs();
        auto src = limbs();

        if (bit_shift == 0) {
            for (size_t i = 0; i < new_len; ++i) {
                rl[i] = src[i + limb_shift];
            }
        } else {
            for (size_t i = 0; i < new_len; ++i) {
                uint64_t low = src[i + limb_shift] >> bit_shift;
                uint64_t high = (i + limb_shift + 1 < size_)
                                    ? (src[i + limb_shift + 1] << (64 - bit_shift))
                                    : 0ULL;
                rl[i] = low | high;
            }
        }

        res.normalize();
        return res;
    }

    [[nodiscard]] BigInt operator&(const BigInt& rhs) const {
        if (!is_negative() && !rhs.is_negative()) {
            return magnitude_and(*this, rhs);
        }
        if (is_negative() && rhs.is_negative()) {
            return ~((~*this) | (~rhs));
        }
        if (!is_negative() && rhs.is_negative()) {
            return magnitude_and_not(*this, ~rhs);
        }
        return rhs & *this;
    }

    [[nodiscard]] BigInt operator|(const BigInt& rhs) const {
        if (!is_negative() && !rhs.is_negative()) {
            return magnitude_or(*this, rhs);
        }
        return ~((~*this) & (~rhs));
    }

    [[nodiscard]] BigInt operator^(const BigInt& rhs) const {
        if (!is_negative() && !rhs.is_negative()) {
            return magnitude_xor(*this, rhs);
        }
        if (is_negative() && rhs.is_negative()) {
            return magnitude_xor(~*this, ~rhs);
        }
        if (is_negative()) {
            return ~(magnitude_xor(~*this, rhs));
        }
        return ~(magnitude_xor(*this, ~rhs));
    }

    // Modular arithmetic
    [[nodiscard]] BigInt mod_pow(const BigInt& exp, const BigInt& modulus) const;
    [[nodiscard]] BigInt mod_inverse(const BigInt& modulus) const;

    // Formatting
    [[nodiscard]] std::string to_hex() const {
        if (is_zero()) {
            return "0x0";
        }
        std::string hex_str;
        auto l = limbs();
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%llx", static_cast<unsigned long long>(l[size_ - 1]));
        hex_str += buf;

        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(size_) - 2; i >= 0; --i) {
            std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(l[i]));
            hex_str += buf;
        }

        std::string res;
        if (negative_) {
            res += "-0x";
        } else {
            res += "0x";
        }
        res += hex_str;
        return res;
    }

    [[nodiscard]] std::string to_string(int base = 10) const {
        if (base < 2 || base > 36) {
            throw std::invalid_argument("Base must be between 2 and 36");
        }
        if (is_zero()) {
            return "0";
        }

        if (base == 10) {
            constexpr uint64_t kBase10Chunk = 1'000'000'000'000'000'000ULL; // 10^18
            std::vector<uint64_t> chunks;
            BigInt temp = *this;
            temp.negative_ = false;

            while (!temp.is_zero()) {
                BigInt q;
                uint64_t rem = divide_single_limb(temp.limbs(), kBase10Chunk, q);
                chunks.push_back(rem);
                temp = std::move(q);
            }

            std::string out;
            if (negative_) {
                out.push_back('-');
            }

            out += std::to_string(chunks.back());
            char buf[32];
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(chunks.size()) - 2; i >= 0; --i) {
                std::snprintf(buf, sizeof(buf), "%018llu",
                              static_cast<unsigned long long>(chunks[i]));
                out += buf;
            }
            return out;
        }

        if (base == 16) {
            return to_hex();
        }

        // Arbitrary base 2..36
        static constexpr char kDigits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
        std::string res;
        BigInt cur = *this;
        cur.negative_ = false;
        BigInt b(base);
        while (!cur.is_zero()) {
            BigInt q, r;
            divide_knuth(cur, b, q, r);
            uint64_t d = r.is_zero() ? 0 : r.limbs()[0];
            res.push_back(kDigits[d]);
            cur = std::move(q);
        }
        if (negative_) {
            res.push_back('-');
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    // Montgomery modular multiplication
    struct MontgomeryContext;
    static MontgomeryContext build_montgomery_context(const BigInt& modulus);
    [[nodiscard]] static BigInt montgomery_multiply(const BigInt& a, const BigInt& b,
                                                    const MontgomeryContext& ctx);

private:
    [[nodiscard]] static BigInt magnitude_and(const BigInt& a, const BigInt& b) {
        size_t min_len = std::min(a.size_, b.size_);
        BigInt res;
        res.allocate_exact(min_len);
        auto rl = res.limbs();
        auto l1 = a.limbs();
        auto l2 = b.limbs();
        for (size_t i = 0; i < min_len; ++i) {
            rl[i] = l1[i] & l2[i];
        }
        res.normalize();
        return res;
    }

    [[nodiscard]] static BigInt magnitude_and_not(const BigInt& a, const BigInt& b) {
        BigInt res;
        res.allocate_exact(a.size_);
        auto rl = res.limbs();
        auto l1 = a.limbs();
        auto l2 = b.limbs();
        for (size_t i = 0; i < a.size_; ++i) {
            uint64_t v2 = (i < b.size_) ? l2[i] : 0ULL;
            rl[i] = l1[i] & (~v2);
        }
        res.normalize();
        return res;
    }

    [[nodiscard]] static BigInt magnitude_or(const BigInt& a, const BigInt& b) {
        size_t max_len = std::max(a.size_, b.size_);
        BigInt res;
        res.allocate_exact(max_len);
        auto rl = res.limbs();
        auto l1 = a.limbs();
        auto l2 = b.limbs();
        for (size_t i = 0; i < max_len; ++i) {
            uint64_t v1 = (i < a.size_) ? l1[i] : 0ULL;
            uint64_t v2 = (i < b.size_) ? l2[i] : 0ULL;
            rl[i] = v1 | v2;
        }
        res.normalize();
        return res;
    }

    [[nodiscard]] static BigInt magnitude_xor(const BigInt& a, const BigInt& b) {
        size_t max_len = std::max(a.size_, b.size_);
        BigInt res;
        res.allocate_exact(max_len);
        auto rl = res.limbs();
        auto l1 = a.limbs();
        auto l2 = b.limbs();
        for (size_t i = 0; i < max_len; ++i) {
            uint64_t v1 = (i < a.size_) ? l1[i] : 0ULL;
            uint64_t v2 = (i < b.size_) ? l2[i] : 0ULL;
            rl[i] = v1 ^ v2;
        }
        res.normalize();
        return res;
    }
    bool negative_{false};
    size_t size_{0};

    // SVO: inline limbs for small BigInts
    uint64_t svo_limbs_[kSVOLimbs]{};
    std::vector<uint64_t> heap_limbs_;

    [[nodiscard]] bool is_svo() const noexcept {
        return size_ <= kSVOLimbs;
    }

    [[nodiscard]] std::span<uint64_t> limbs() noexcept {
        return is_svo() ? std::span<uint64_t>(svo_limbs_, size_)
                        : std::span<uint64_t>(heap_limbs_.data(), size_);
    }

    [[nodiscard]] std::span<const uint64_t> limbs() const noexcept {
        return is_svo() ? std::span<const uint64_t>(svo_limbs_, size_)
                        : std::span<const uint64_t>(heap_limbs_.data(), size_);
    }

    void allocate_exact(size_t new_limbs) {
        if (new_limbs <= kSVOLimbs) {
            std::memset(svo_limbs_, 0, sizeof(svo_limbs_));
            heap_limbs_.clear();
            heap_limbs_.shrink_to_fit();
        } else {
            heap_limbs_.assign(new_limbs, 0);
        }
        size_ = new_limbs;
    }

    void normalize() noexcept {
        auto l = limbs();
        while (size_ > 0 && l[size_ - 1] == 0) {
            --size_;
        }
        if (size_ == 0) {
            negative_ = false;
        }
        if (size_ <= kSVOLimbs && !heap_limbs_.empty()) {
            std::memcpy(svo_limbs_, heap_limbs_.data(), size_ * sizeof(uint64_t));
            heap_limbs_.clear();
            heap_limbs_.shrink_to_fit();
        }
    }

    static std::strong_ordering compare_magnitudes(std::span<const uint64_t> a,
                                                   std::span<const uint64_t> b) noexcept {
        size_t a_size = a.size();
        while (a_size > 0 && a[a_size - 1] == 0) {
            --a_size;
        }
        size_t b_size = b.size();
        while (b_size > 0 && b[b_size - 1] == 0) {
            --b_size;
        }
        if (a_size != b_size) {
            return a_size <=> b_size;
        }
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(a_size) - 1; i >= 0; --i) {
            if (a[i] != b[i]) {
                return a[i] <=> b[i];
            }
        }
        return std::strong_ordering::equal;
    }

    static BigInt add_magnitudes(std::span<const uint64_t> a, std::span<const uint64_t> b) {
        size_t max_len = std::max(a.size(), b.size());
        BigInt res;
        res.allocate_exact(max_len + 1);
        auto rl = res.limbs();

        unsigned char carry = 0;
        for (size_t i = 0; i < max_len; ++i) {
            uint64_t ai = (i < a.size()) ? a[i] : 0;
            uint64_t bi = (i < b.size()) ? b[i] : 0;
            carry = detail::add_carry_u64(carry, ai, bi, &rl[i]);
        }
        rl[max_len] = carry;
        res.normalize();
        return res;
    }

    // Assumes a >= b
    static BigInt sub_magnitudes(std::span<const uint64_t> a, std::span<const uint64_t> b) {
        BigInt res;
        res.allocate_exact(a.size());
        auto rl = res.limbs();

        unsigned char borrow = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            uint64_t ai = a[i];
            uint64_t bi = (i < b.size()) ? b[i] : 0;
            borrow = detail::sub_borrow_u64(borrow, ai, bi, &rl[i]);
        }
        res.normalize();
        return res;
    }

    // Single-limb division fast path
    static uint64_t divide_single_limb(std::span<const uint64_t> u, uint64_t v, BigInt& q) {
        assert(v != 0);
        q.allocate_exact(u.size());
        auto ql = q.limbs();
        uint64_t rem = 0;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(u.size()) - 1; i >= 0; --i) {
            ql[i] = detail::div128_64(rem, u[i], v, &rem);
        }
        q.normalize();
        return rem;
    }

    // Multi-precision Knuth Algorithm D division
    static void divide_knuth(const BigInt& u, const BigInt& v, BigInt& q, BigInt& r) {
        if (v.is_zero()) {
            throw std::invalid_argument("Division by zero in BigInt");
        }
        if (u.is_zero()) {
            q = BigInt();
            r = BigInt();
            return;
        }

        std::strong_ordering cmp = compare_magnitudes(u.limbs(), v.limbs());
        if (cmp == std::strong_ordering::less) {
            q = BigInt();
            r = u;
            return;
        }
        if (cmp == std::strong_ordering::equal) {
            q = BigInt(1LL);
            q.negative_ = (u.negative_ != v.negative_);
            r = BigInt();
            return;
        }

        if (v.size_ == 1) {
            uint64_t rem = divide_single_limb(u.limbs(), v.limbs()[0], q);
            q.negative_ = (u.negative_ != v.negative_);
            r = BigInt(rem);
            r.negative_ = u.negative_;
            return;
        }

        size_t u_len = u.size_;
        size_t v_len = v.size_;
        size_t n = v_len;
        size_t m = u_len - n;

        // Step D1: Normalize
        int norm_shift = std::countl_zero(v.limbs()[n - 1]);
        std::vector<uint64_t> vn(n, 0);
        std::vector<uint64_t> un(m + n + 1, 0);

        if (norm_shift == 0) {
            std::memcpy(vn.data(), v.limbs().data(), n * sizeof(uint64_t));
            std::memcpy(un.data(), u.limbs().data(), u_len * sizeof(uint64_t));
            un[m + n] = 0;
        } else {
            uint64_t carry = 0;
            for (size_t i = 0; i < n; ++i) {
                vn[i] = (v.limbs()[i] << norm_shift) | carry;
                carry = v.limbs()[i] >> (64 - norm_shift);
            }

            carry = 0;
            for (size_t i = 0; i < u_len; ++i) {
                un[i] = (u.limbs()[i] << norm_shift) | carry;
                carry = u.limbs()[i] >> (64 - norm_shift);
            }
            un[m + n] = carry;
        }

        q.allocate_exact(m + 1);
        auto ql = q.limbs();

        // Step D2: Loop j from m down to 0
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(m); j >= 0; --j) {
            // Step D3: Calculate trial quotient q_hat
            uint64_t q_hat = 0;
            uint64_t r_hat = 0;
            bool r_overflow = false;

            if (un[j + n] == vn[n - 1]) {
                q_hat = UINT64_MAX;
                r_hat = un[j + n - 1] + vn[n - 1];
                r_overflow = (r_hat < vn[n - 1]);
            } else {
                q_hat = detail::div128_64(un[j + n], un[j + n - 1], vn[n - 1], &r_hat);
                r_overflow = false;
            }

            while (!r_overflow) {
                uint64_t p_hi = 0;
                uint64_t p_lo = detail::mulx_u64(q_hat, vn[n - 2], &p_hi);
                if (p_hi > r_hat || (p_hi == r_hat && p_lo > un[j + n - 2])) {
                    --q_hat;
                    unsigned char c = detail::add_carry_u64(0, r_hat, vn[n - 1], &r_hat);
                    if (c != 0) {
                        r_overflow = true;
                    }
                } else {
                    break;
                }
            }

            // Step D4: Multiply and subtract
            uint64_t carry = 0;
            for (size_t i = 0; i < n; ++i) {
                uint64_t p_hi = 0;
                uint64_t p_lo = detail::mulx_u64(q_hat, vn[i], &p_hi);
                unsigned char c = detail::add_carry_u64(0, p_lo, carry, &p_lo);
                p_hi += c;
                carry = p_hi;

                unsigned char b = detail::sub_borrow_u64(0, un[j + i], p_lo, &un[j + i]);
                carry += b;
            }

            bool borrow = (un[j + n] < carry);
            un[j + n] -= carry;

            // Step D5: Add-back if borrow
            if (borrow) {
                --q_hat;
                unsigned char add_carry = 0;
                for (size_t i = 0; i < n; ++i) {
                    add_carry = detail::add_carry_u64(add_carry, un[j + i], vn[i], &un[j + i]);
                }
                un[j + n] += add_carry;
            }

            ql[j] = q_hat;
        }

        // Step D7: Unnormalize remainder
        r.allocate_exact(n);
        auto rl = r.limbs();
        if (norm_shift == 0) {
            std::memcpy(rl.data(), un.data(), n * sizeof(uint64_t));
        } else {
            uint64_t carry = 0;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
                rl[i] = (un[i] >> norm_shift) | carry;
                carry = un[i] << (64 - norm_shift);
            }
        }

        q.negative_ = (u.negative_ != v.negative_);
        q.normalize();
        r.negative_ = u.negative_;
        r.normalize();
    }

public:
    // Multiplication strategy selection
    [[nodiscard]] static BigInt multiply_schoolbook(std::span<const uint64_t> a,
                                                    std::span<const uint64_t> b) {
        if (a.empty() || b.empty()) {
            return BigInt();
        }
        size_t n = a.size();
        size_t m = b.size();
        BigInt res;
        res.allocate_exact(n + m);
        detail::multiply_schoolbook_raw(a.data(), n, b.data(), m, res.limbs().data());
        res.normalize();
        return res;
    }

    // Subtractive Karatsuba Multiplication O(n^1.585)
    [[nodiscard]] static BigInt multiply_karatsuba(std::span<const uint64_t> a,
                                                   std::span<const uint64_t> b) {
        if (a.empty() || b.empty()) {
            return BigInt();
        }
        size_t n = std::max(a.size(), b.size());
        if (n <= kKaratsubaThreshold) {
            return multiply_schoolbook(a, b);
        }

        const uint64_t* a_ptr = a.data();
        const uint64_t* b_ptr = b.data();
        alignas(64) uint64_t stack_a[64];
        alignas(64) uint64_t stack_b[64];
        std::vector<uint64_t> heap_a, heap_b;

        if (a.size() < n || b.size() < n) {
            if (n <= 64) {
                std::memset(stack_a, 0, n * sizeof(uint64_t));
                std::memset(stack_b, 0, n * sizeof(uint64_t));
                std::memcpy(stack_a, a.data(), a.size() * sizeof(uint64_t));
                std::memcpy(stack_b, b.data(), b.size() * sizeof(uint64_t));
                a_ptr = stack_a;
                b_ptr = stack_b;
            } else {
                heap_a.assign(n, 0);
                heap_b.assign(n, 0);
                std::memcpy(heap_a.data(), a.data(), a.size() * sizeof(uint64_t));
                std::memcpy(heap_b.data(), b.data(), b.size() * sizeof(uint64_t));
                a_ptr = heap_a.data();
                b_ptr = heap_b.data();
            }
        }

        BigInt res;
        res.allocate_exact(2 * n);
        auto rl = res.limbs();

        if (n == 8) {
            detail::karatsuba_8x8(a_ptr, b_ptr, rl.data());
            res.normalize();
            return res;
        }

        alignas(64) uint64_t stack_scratch[512];
        std::vector<uint64_t> heap_scratch;
        uint64_t* scratch_ptr = stack_scratch;
        size_t needed_scratch = 12 * n;
        if (needed_scratch > 512) {
            heap_scratch.resize(needed_scratch);
            scratch_ptr = heap_scratch.data();
        }

        detail::karatsuba_raw(a_ptr, b_ptr, n, rl.data(), scratch_ptr);
        res.normalize();
        return res;
    }

    [[nodiscard]] static BigInt multiply_schoolbook(const BigInt& a, const BigInt& b) {
        BigInt res = multiply_schoolbook(a.limbs(), b.limbs());
        res.negative_ = (a.negative_ != b.negative_);
        res.normalize();
        return res;
    }

    [[nodiscard]] static BigInt multiply_karatsuba(const BigInt& a, const BigInt& b) {
        BigInt res = multiply_karatsuba(a.limbs(), b.limbs());
        res.negative_ = (a.negative_ != b.negative_);
        res.normalize();
        return res;
    }
};

struct BigInt::MontgomeryContext {
    BigInt modulus;
    size_t k{0};         // Limb count of modulus
    uint64_t n_prime{0}; // -N_0^(-1) mod 2^64
    BigInt r_squared;    // (2^(128k)) mod modulus
};

inline BigInt::MontgomeryContext BigInt::build_montgomery_context(const BigInt& modulus) {
    if (modulus <= BigInt(0LL) || (modulus.limbs()[0] & 1ULL) == 0ULL) {
        throw std::invalid_argument("Montgomery context requires a strictly positive, odd modulus");
    }
    MontgomeryContext ctx;
    ctx.modulus = modulus;
    ctx.k = modulus.size_;
    ctx.n_prime = detail::compute_montgomery_n_prime(modulus.limbs()[0]);

    // R^2 mod modulus: R = 2^(64k), R^2 = 2^(128k)
    BigInt r2 = BigInt(1LL) << (128 * ctx.k);
    ctx.r_squared = r2 % modulus;
    return ctx;
}

inline BigInt BigInt::montgomery_multiply(const BigInt& a, const BigInt& b,
                                          const BigInt::MontgomeryContext& ctx) {
    size_t k = ctx.k;
    std::vector<uint64_t> t(k + 2, 0);

    auto al = a.limbs();
    auto bl = b.limbs();
    auto nl = ctx.modulus.limbs();

    for (size_t i = 0; i < k; ++i) {
        uint64_t bi = (i < b.size_) ? bl[i] : 0;
        uint64_t carry = 0;
        for (size_t j = 0; j < k; ++j) {
            uint64_t aj = (j < a.size_) ? al[j] : 0;
            carry = detail::mul_add_carry(aj, bi, carry, &t[j]);
        }
        unsigned char c1 = detail::add_carry_u64(0, t[k], carry, &t[k]);
        detail::add_carry_u64(c1, t[k + 1], 0, &t[k + 1]);

        uint64_t m = t[0] * ctx.n_prime;
        uint64_t dummy = t[0];
        carry = detail::mul_add_carry(m, nl[0], 0, &dummy);

        for (size_t j = 1; j < k; ++j) {
            uint64_t tj = t[j];
            carry = detail::mul_add_carry(m, nl[j], carry, &tj);
            t[j - 1] = tj;
        }

        uint64_t tk = t[k];
        unsigned char c_red = detail::add_carry_u64(0, tk, carry, &t[k - 1]);
        unsigned char c_red1 = detail::add_carry_u64(c_red, t[k + 1], 0, &t[k]);
        t[k + 1] = c_red1;
    }

    bool ge = false;
    if (t[k + 1] > 0 || t[k] > 0) {
        ge = true;
    } else {
        for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(k) - 1; j >= 0; --j) {
            if (t[j] > nl[j]) {
                ge = true;
                break;
            }
            if (t[j] < nl[j]) {
                ge = false;
                break;
            }
            if (j == 0) {
                ge = true;
            }
        }
    }

    BigInt out;
    out.allocate_exact(k);
    auto ol = out.limbs();

    if (ge) {
        unsigned char borrow = 0;
        for (size_t j = 0; j < k; ++j) {
            borrow = detail::sub_borrow_u64(borrow, t[j], nl[j], &ol[j]);
        }
    } else {
        for (size_t j = 0; j < k; ++j) {
            ol[j] = t[j];
        }
    }

    out.normalize();
    return out;
}

inline BigInt BigInt::mod_pow(const BigInt& exp, const BigInt& modulus) const {
    if (modulus.is_zero()) {
        throw std::invalid_argument("mod_pow: modulus cannot be zero");
    }
    BigInt mod_pos = modulus;
    mod_pos.negative_ = false;
    if (mod_pos == BigInt(1LL)) {
        return BigInt();
    }

    if (exp < BigInt(0LL)) {
        BigInt base_inv = this->mod_inverse(mod_pos);
        return base_inv.mod_pow(-exp, mod_pos);
    }

    if (exp.is_zero()) {
        return BigInt(1LL) % mod_pos;
    }

    if (is_zero()) {
        return BigInt();
    }

    BigInt base_val = *this % mod_pos;
    if (base_val.is_negative()) {
        base_val += mod_pos;
    }

    // Use Montgomery representation when modulus is odd
    if ((mod_pos.limbs()[0] & 1ULL) != 0ULL) {
        MontgomeryContext ctx = build_montgomery_context(mod_pos);
        BigInt a_hat = montgomery_multiply(base_val, ctx.r_squared, ctx);
        BigInt c_hat = montgomery_multiply(BigInt(1LL), ctx.r_squared, ctx);

        size_t total_bits = exp.num_bits();
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(total_bits) - 1; i >= 0; --i) {
            c_hat = montgomery_multiply(c_hat, c_hat, ctx);
            size_t limb_idx = static_cast<size_t>(i) / 64;
            size_t bit_idx = static_cast<size_t>(i) % 64;
            if ((exp.limbs()[limb_idx] >> bit_idx) & 1ULL) {
                c_hat = montgomery_multiply(c_hat, a_hat, ctx);
            }
        }
        return montgomery_multiply(c_hat, BigInt(1LL), ctx);
    }

    // Fallback standard square-and-multiply for even modulus
    BigInt result = BigInt(1LL);
    BigInt cur_base = base_val;
    size_t total_bits = exp.num_bits();
    for (size_t i = 0; i < total_bits; ++i) {
        size_t limb_idx = i / 64;
        size_t bit_idx = i % 64;
        if ((exp.limbs()[limb_idx] >> bit_idx) & 1ULL) {
            result = (result * cur_base) % mod_pos;
        }
        cur_base = (cur_base * cur_base) % mod_pos;
    }
    return result;
}

inline BigInt BigInt::mod_inverse(const BigInt& modulus) const {
    if (modulus <= BigInt(0LL)) {
        throw std::invalid_argument("mod_inverse: modulus must be strictly positive");
    }
    BigInt a = *this % modulus;
    if (a < BigInt(0LL))
        a += modulus;
    if (a.is_zero()) {
        throw std::invalid_argument("mod_inverse: modular inverse does not exist");
    }

    BigInt m0 = modulus;
    BigInt y(0LL);
    BigInt x(1LL);
    if (modulus == BigInt(1LL)) {
        return BigInt(0LL);
    }

    BigInt temp_a = a;
    BigInt temp_m = modulus;

    while (temp_a > BigInt(1LL)) {
        if (temp_m.is_zero()) {
            throw std::invalid_argument("mod_inverse: modular inverse does not exist");
        }
        BigInt q = temp_a / temp_m;
        BigInt t = temp_m;

        temp_m = temp_a % temp_m;
        temp_a = t;
        t = y;

        y = x - q * y;
        x = t;
    }

    if (temp_a != BigInt(1LL)) {
        throw std::invalid_argument("mod_inverse: modular inverse does not exist");
    }

    if (x < BigInt(0LL)) {
        x += modulus;
    }
    return x;
}

} // namespace algoat::numerics
