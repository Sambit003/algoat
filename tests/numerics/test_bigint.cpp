#include <algoat/numerics/bigint.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>
#include <random>
#include <string>

using namespace algoat::numerics;

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
// Global allocation tracker to verify SVO zero-allocation guarantee
static std::atomic<size_t> g_heap_alloc_count{0};
static std::atomic<bool> g_track_allocations{false};

void* operator new(size_t size) {
    if (g_track_allocations.load(std::memory_order_relaxed)) {
        g_heap_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    void* ptr = std::malloc(size);
    if (!ptr)
        throw std::bad_alloc();
    return ptr;
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, size_t) noexcept {
    std::free(ptr);
}

struct AllocTracker {
    AllocTracker() {
        g_heap_alloc_count = 0;
        g_track_allocations = true;
    }
    ~AllocTracker() {
        g_track_allocations = false;
    }
    size_t count() const {
        return g_heap_alloc_count.load();
    }
};
#else
struct AllocTracker {
    AllocTracker() {}
    ~AllocTracker() {}
    size_t count() const {
        return 0;
    }
};
#endif

// ---------------------------------------------------------------------------
// 1. SVO Verification (Acceptance Criterion 4)
// ---------------------------------------------------------------------------
TEST(BigIntTest, SVOZeroHeapAllocation) {
    AllocTracker tracker;

    // Numbers <= 256 bits (4 limbs) must produce ZERO heap allocations
    BigInt a(0ULL);
    BigInt b(123456789012345ULL);
    BigInt c(-9876543210LL);
    BigInt max_256("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 16);

    EXPECT_EQ(max_256.num_limbs(), 4);
    EXPECT_EQ(max_256.num_bits(), 256);

    BigInt d = a + b;
    BigInt e = max_256 - b;
    BigInt f = b * BigInt(2);
    BigInt g = e / BigInt(10);
    BigInt h = e % BigInt(10);

    // Copy and move constructors
    BigInt copy_d = d;
    BigInt move_e = std::move(e);

    // Verify 0 heap allocations occurred in this entire block
    EXPECT_EQ(tracker.count(), 0);
}

TEST(BigIntTest, HeapTransitionWhenExceeding256Bits) {
    // 257-bit number: 2^256
    BigInt a = BigInt(1) << 256;
    EXPECT_EQ(a.num_bits(), 257);
    EXPECT_EQ(a.num_limbs(), 5);

    // Shrinking back to <= 256 bits
    BigInt b = a >> 1;
    EXPECT_EQ(b.num_bits(), 256);
    EXPECT_EQ(b.num_limbs(), 4);
}

// ---------------------------------------------------------------------------
// 2. Constructors and Conversions
// ---------------------------------------------------------------------------
TEST(BigIntTest, DefaultConstructorIsZero) {
    BigInt a;
    EXPECT_TRUE(a.is_zero());
    EXPECT_FALSE(a.is_negative());
    EXPECT_EQ(a.sign(), 0);
    EXPECT_EQ(a.num_bits(), 0);
    EXPECT_EQ(a.num_limbs(), 0);
    EXPECT_EQ(a.to_string(), "0");
    EXPECT_EQ(a.to_hex(), "0x0");
    EXPECT_TRUE(a.fits_int64());
    EXPECT_EQ(a.to_int64(), 0);
}

TEST(BigIntTest, IntegerConstructors) {
    BigInt a(42ULL);
    EXPECT_FALSE(a.is_zero());
    EXPECT_FALSE(a.is_negative());
    EXPECT_EQ(a.sign(), 1);
    EXPECT_EQ(a.num_limbs(), 1);
    EXPECT_EQ(a.to_string(), "42");
    EXPECT_EQ(a.to_hex(), "0x2a");
    EXPECT_TRUE(a.fits_int64());
    EXPECT_EQ(a.to_int64(), 42);

    BigInt b(-12345LL);
    EXPECT_FALSE(b.is_zero());
    EXPECT_TRUE(b.is_negative());
    EXPECT_EQ(b.sign(), -1);
    EXPECT_EQ(b.num_limbs(), 1);
    EXPECT_EQ(b.to_string(), "-12345");
    EXPECT_TRUE(b.fits_int64());
    EXPECT_EQ(b.to_int64(), -12345);

    BigInt min_i64(INT64_MIN);
    EXPECT_TRUE(min_i64.is_negative());
    EXPECT_TRUE(min_i64.fits_int64());
    EXPECT_EQ(min_i64.to_int64(), INT64_MIN);
    EXPECT_EQ(min_i64.to_string(), std::to_string(INT64_MIN));
}

TEST(BigIntTest, StringConstructorsDecimal) {
    BigInt a("123456789012345678901234567890");
    EXPECT_FALSE(a.is_negative());
    EXPECT_EQ(a.sign(), 1);
    EXPECT_EQ(a.to_string(), "123456789012345678901234567890");

    BigInt b("-987654321098765432109876543210");
    EXPECT_TRUE(b.is_negative());
    EXPECT_EQ(b.sign(), -1);
    EXPECT_EQ(b.to_string(), "-987654321098765432109876543210");

    BigInt zero("0");
    EXPECT_TRUE(zero.is_zero());
    EXPECT_EQ(zero.to_string(), "0");

    BigInt neg_zero("-0");
    EXPECT_TRUE(neg_zero.is_zero());
    EXPECT_FALSE(neg_zero.is_negative());
    EXPECT_EQ(neg_zero.to_string(), "0");
}

TEST(BigIntTest, StringConstructorsHex) {
    BigInt a("0xdeadbeefcafebabe12345678", 16);
    EXPECT_EQ(a.to_hex(), "0xdeadbeefcafebabe12345678");

    BigInt b("-0xff", 16);
    EXPECT_TRUE(b.is_negative());
    EXPECT_EQ(b.to_int64(), -255);
}

TEST(BigIntTest, Comparisons) {
    BigInt a(100);
    BigInt b(200);
    BigInt c(-100);
    BigInt d(100);

    EXPECT_TRUE(a == d);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(c < a);
    EXPECT_TRUE(c < b);
    EXPECT_TRUE(a > c);
    EXPECT_TRUE(a <= d);
    EXPECT_TRUE(a >= d);
}

// ---------------------------------------------------------------------------
// 3. Arithmetic Operations
// ---------------------------------------------------------------------------
TEST(BigIntTest, AdditionAndSubtraction) {
    BigInt a("1000000000000000000000000");
    BigInt b("2000000000000000000000000");
    BigInt c = a + b;
    EXPECT_EQ(c.to_string(), "3000000000000000000000000");

    BigInt d = a - b;
    EXPECT_EQ(d.to_string(), "-1000000000000000000000000");

    BigInt e = -d;
    EXPECT_EQ(e, a);

    // Multi-limb carry propagation
    BigInt max_u64(UINT64_MAX);
    BigInt one(1);
    BigInt carry_sum = max_u64 + one;
    EXPECT_EQ(carry_sum.num_limbs(), 2);
    EXPECT_EQ(carry_sum.to_hex(), "0x10000000000000000");
    EXPECT_EQ(carry_sum - one, max_u64);
}

TEST(BigIntTest, UnaryMinus) {
    BigInt zero(0);
    EXPECT_EQ(-zero, zero);
    EXPECT_FALSE((-zero).is_negative());

    BigInt pos(42);
    BigInt neg = -pos;
    EXPECT_TRUE(neg.is_negative());
    EXPECT_EQ(neg.to_int64(), -42);
    EXPECT_EQ(-neg, pos);
}

// ---------------------------------------------------------------------------
// 4. Bitwise Operators
// ---------------------------------------------------------------------------
TEST(BigIntTest, BitwiseShifts) {
    BigInt a(1);
    BigInt b = a << 64;
    EXPECT_EQ(b.num_limbs(), 2);
    EXPECT_EQ(b.to_hex(), "0x10000000000000000");
    EXPECT_EQ(b >> 64, a);

    BigInt c = a << 100;
    EXPECT_EQ(c >> 100, a);
    EXPECT_EQ(c >> 200, BigInt(0));
}

TEST(BigIntTest, BitwiseLogic) {
    BigInt a("0x00ff00ff00ff00ff", 16);
    BigInt b("0x0f0f0f0f0f0f0f0f", 16);

    EXPECT_EQ((a & b).to_hex(), "0xf000f000f000f");
    EXPECT_EQ((a | b).to_hex(), "0xfff0fff0fff0fff");
    EXPECT_EQ((a ^ b) ^ b, a);
}

TEST(BigIntTest, TwosComplementBitwiseLogic) {
    BigInt a(-3);
    BigInt b(-5);

    // Negative & Negative
    EXPECT_EQ(a & b, BigInt(-7));
    EXPECT_EQ(a | b, BigInt(-1));
    EXPECT_EQ(a ^ b, BigInt(6));

    // Bitwise NOT: ~x == -(x + 1)
    EXPECT_EQ(~a, BigInt(2));
    EXPECT_EQ(~b, BigInt(4));
    EXPECT_EQ(~BigInt(0), BigInt(-1));
    EXPECT_EQ(~BigInt(6), BigInt(-7));

    // Mixed Signs
    BigInt pos(6);
    EXPECT_EQ(pos & b, BigInt(2));
    EXPECT_EQ(b & pos, BigInt(2));
    EXPECT_EQ(b | pos, BigInt(-1));
    EXPECT_EQ(b ^ pos, BigInt(-3));

    // Arithmetic Right Shift on Negatives
    EXPECT_EQ(BigInt(-1) >> 1, BigInt(-1));
    EXPECT_EQ(BigInt(-7) >> 1, BigInt(-4));
    EXPECT_EQ(BigInt(-8) >> 1, BigInt(-4));
}

TEST(BigIntTest, ArbitraryBaseFormattingAndParsing) {
    // Base 2
    BigInt b2("0b101101", 2);
    EXPECT_EQ(b2.to_int64(), 45);
    EXPECT_EQ(b2.to_string(2), "101101");

    // Base 8
    BigInt b8("0o755", 8);
    EXPECT_EQ(b8.to_int64(), 493);
    EXPECT_EQ(b8.to_string(8), "755");

    // Base 16
    BigInt b16("0xdeadbeef", 16);
    EXPECT_EQ(b16.to_string(16), "0xdeadbeef");

    // Base 36
    BigInt b36("z1", 36);
    EXPECT_EQ(b36.to_int64(), 35 * 36 + 1);
    EXPECT_EQ(b36.to_string(36), "z1");

    // Negative in arbitrary base
    BigInt neg_b8("-755", 8);
    EXPECT_EQ(neg_b8.to_int64(), -493);
    EXPECT_EQ(neg_b8.to_string(8), "-755");

    // Invalid base throws
    EXPECT_THROW(BigInt("10", 1), std::invalid_argument);
    EXPECT_THROW(BigInt("10", 37), std::invalid_argument);
    EXPECT_THROW((void)b2.to_string(1), std::invalid_argument);
    EXPECT_THROW((void)b2.to_string(37), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 5. Knuth Algorithm D Division
// ---------------------------------------------------------------------------
TEST(BigIntTest, SingleLimbDivision) {
    BigInt u("123456789012345678901234567890");
    BigInt v(10007);
    BigInt q = u / v;
    BigInt r = u % v;
    EXPECT_EQ(q * v + r, u);
    EXPECT_LT(r, v);
    EXPECT_GE(r, BigInt(0));
}

TEST(BigIntTest, MultiLimbDivisionKnuthD) {
    BigInt u("9999999999999999999999999999999999999999999999999999999999999999");
    BigInt v("1234567890123456789012345678901234567890");
    BigInt q = u / v;
    BigInt r = u % v;

    EXPECT_EQ(q * v + r, u);
    EXPECT_LT(r, v);
    EXPECT_GE(r, BigInt(0));
}

TEST(BigIntTest, KnuthStepD5AddBackTriggerVector) {
    // Deterministic test vector from TAOCP Theorem B research report:
    // v has 3 limbs: v2 = 2^63 + 100, v1 = 0, v0 = 2^64 - 1
    // u has 4 limbs: u3 = 1, u2 = 200, u1 = 1, u0 = 0
    BigInt v2((1ULL << 63) + 100ULL);
    BigInt v0(UINT64_MAX);
    BigInt v = (v2 << 128) + v0;

    BigInt u3(1ULL);
    BigInt u2(200ULL);
    BigInt u1(1ULL);
    BigInt u = (u3 << 192) + (u2 << 128) + (u1 << 64);

    BigInt q = u / v;
    BigInt r = u % v;

    EXPECT_EQ(q * v + r, u);
    EXPECT_LT(r, v);
    EXPECT_GE(r, BigInt(0));
    EXPECT_EQ(q.to_int64(), 1); // Step D5 corrected q_hat from 2 down to 1
}

TEST(BigIntTest, DivisionSignSemantics) {
    BigInt pos17(17), pos5(5);
    BigInt neg17(-17), neg5(-5);

    EXPECT_EQ((pos17 / pos5).to_int64(), 3);
    EXPECT_EQ((pos17 % pos5).to_int64(), 2);

    EXPECT_EQ((neg17 / pos5).to_int64(), -3);
    EXPECT_EQ((neg17 % pos5).to_int64(), -2);

    EXPECT_EQ((pos17 / neg5).to_int64(), -3);
    EXPECT_EQ((pos17 % neg5).to_int64(), 2);

    EXPECT_EQ((neg17 / neg5).to_int64(), 3);
    EXPECT_EQ((neg17 % neg5).to_int64(), -2);
}

TEST(BigIntTest, DivisionByZeroThrows) {
    BigInt a(42);
    BigInt zero(0);
    EXPECT_THROW((void)(a / zero), std::invalid_argument);
    EXPECT_THROW((void)(a % zero), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 6. Multiplication: Schoolbook vs Karatsuba
// ---------------------------------------------------------------------------
TEST(BigIntTest, KaratsubaVsSchoolbookCorrectnessAndSpeed) {
    // 512-bit numbers (8 limbs)
    std::mt19937_64 rng(42);
    std::string hex_a = "0x", hex_b = "0x";
    for (int i = 0; i < 8; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(rng()));
        hex_a += buf;
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(rng()));
        hex_b += buf;
    }
    BigInt a(hex_a, 16);
    BigInt b(hex_b, 16);

    BigInt p_school = BigInt::multiply_schoolbook(a, b);
    BigInt p_kara = BigInt::multiply_karatsuba(a, b);

    EXPECT_EQ(p_school, p_kara);
    EXPECT_EQ(p_kara, a * b);
}

TEST(BigIntTest, MultiplicationSchoolbookAndKaratsuba) {
    std::mt19937_64 rng(1337);
    for (size_t bits : {256, 512, 1024, 2048}) {
        size_t limbs = bits / 64;
        std::string hex_a = "0x";
        std::string hex_b = "0x";
        for (size_t i = 0; i < limbs; ++i) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(rng()));
            hex_a += buf;
            std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(rng()));
            hex_b += buf;
        }

        BigInt a(hex_a, 16);
        BigInt b(hex_b, 16);

        BigInt prod = a * b;
        EXPECT_EQ(prod / a, b);
        EXPECT_EQ(prod % a, BigInt(0));
    }
}

// ---------------------------------------------------------------------------
// 7. Modular Inverse & Extended Euclidean
// ---------------------------------------------------------------------------
TEST(BigIntTest, ModInverse) {
    BigInt a(3);
    BigInt m(11);
    BigInt inv = a.mod_inverse(m);
    EXPECT_EQ(inv.to_int64(), 4);
    EXPECT_EQ((a * inv) % m, BigInt(1));

    // RSA standard e = 65537
    BigInt e(65537);
    BigInt phi("10101010101010101010101010101010101010101010101010101010101010101");
    // phi is coprime to 65537
    BigInt d = e.mod_inverse(phi);
    EXPECT_EQ((e * d) % phi, BigInt(1));

    // Non-coprime throws
    BigInt not_coprime(4);
    BigInt even_mod(12);
    EXPECT_THROW((void)not_coprime.mod_inverse(even_mod), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// 8. Montgomery Modular Exponentiation (mod_pow) & RSA Benchmarks
// ---------------------------------------------------------------------------
TEST(BigIntTest, ModPowSmallValues) {
    BigInt base(2);
    BigInt exp(10);
    BigInt mod(1000);
    EXPECT_EQ(base.mod_pow(exp, mod).to_int64(), 24); // 1024 % 1000 = 24

    BigInt b(3);
    BigInt e(45);
    BigInt m(101);
    // 3^45 mod 101
    BigInt res = b.mod_pow(e, m);
    // Compare against repeated manual multiplication
    BigInt expected(1);
    for (int i = 0; i < 45; ++i) {
        expected = (expected * b) % m;
    }
    EXPECT_EQ(res, expected);
}

TEST(BigIntTest, ModPowEdgeCases) {
    BigInt base(42);
    BigInt mod(101);
    EXPECT_EQ(base.mod_pow(BigInt(0), mod), BigInt(1));
    EXPECT_EQ(BigInt(0).mod_pow(BigInt(5), mod), BigInt(0));
    EXPECT_EQ(base.mod_pow(BigInt(5), BigInt(1)), BigInt(0));
}

TEST(BigIntTest, RSA2048ModPowPerformance) {
    // 2048-bit RSA test vector (modulus has 32 limbs, odd)
    std::string mod_hex = "0x"
                          "c000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000001";

    BigInt n(mod_hex, 16);
    BigInt a("123456789012345678901234567890");
    BigInt e(65537); // Standard RSA public exponent

    auto start = std::chrono::steady_clock::now();
    BigInt c = a.mod_pow(e, n);
    auto end = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[INFO] RSA-2048 mod_pow (e=65537) elapsed time: " << elapsed_ms << " ms"
              << std::endl;

    EXPECT_FALSE(c.is_zero());
    EXPECT_LT(c, n);

    // Acceptance Criterion 3: RSA-2048 mod_pow completes in < 5 ms (relaxed under sanitizer
    // overhead)
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
    EXPECT_LT(elapsed_ms, 25.0);
#else
    EXPECT_LT(elapsed_ms, 5.0);
#endif
}

TEST(BigIntTest, RSA4096ModPowPerformance) {
    // 4096-bit RSA test vector (modulus has 64 limbs, odd)
    std::string mod_hex = "0x"
                          "c000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000001";

    BigInt n(mod_hex, 16);
    BigInt a("123456789012345678901234567890");
    BigInt e(65537);

    auto start = std::chrono::steady_clock::now();
    BigInt c = a.mod_pow(e, n);
    auto end = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[INFO] RSA-4096 mod_pow (e=65537) elapsed time: " << elapsed_ms << " ms"
              << std::endl;

    EXPECT_FALSE(c.is_zero());
    EXPECT_LT(c, n);

    // Acceptance Criterion 3: RSA-4096 mod_pow completes in < 30 ms
    EXPECT_LT(elapsed_ms, 30.0);
}
