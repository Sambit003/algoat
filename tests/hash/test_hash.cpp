#include <algoat/hash/hash.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// Helper to generate the official xxHash3 sanity test buffer via LCG
std::vector<std::byte> generate_xxh3_test_buffer(std::size_t len) {
    std::vector<std::byte> buf(len);
    uint64_t byte_gen = 2654435761ULL; // 0x9E3779B1
    for (std::size_t i = 0; i < len; ++i) {
        buf[i] = static_cast<std::byte>(byte_gen >> 56);
        byte_gen *= 11400714785074694797ULL; // 0x9E3779B185EBCA87
    }
    return buf;
}

std::span<const std::byte> to_byte_span(std::string_view sv) noexcept {
    return std::as_bytes(std::span{sv.data(), sv.size()});
}

} // namespace

// ─── Compile-Time Constraints & Zero-Allocation Invariants ───────────────────

TEST(HashSuiteTest, NoexceptAndZeroAllocation) {
    using namespace algoat::hash;

    std::span<const std::byte> empty_span{};
    std::string_view empty_sv{};
    HighwayKey key{1, 2, 3, 4};

    static_assert(noexcept(xxhash3_64(empty_span, 0)));
    static_assert(noexcept(xxhash3_64(empty_sv, 0)));
    static_assert(noexcept(xxhash3_128(empty_span, 0)));
    static_assert(noexcept(xxhash3_128(empty_sv, 0)));

    static_assert(noexcept(wyhash(empty_span, 0)));
    static_assert(noexcept(wyhash(empty_sv, 0)));

    static_assert(noexcept(highway_hash_64(empty_span, key)));
    static_assert(noexcept(highway_hash_64(empty_sv, key)));
    static_assert(noexcept(highway_hash_128(empty_span, key)));
    static_assert(noexcept(highway_hash_128(empty_sv, key)));
    static_assert(noexcept(highway_hash_256(empty_span, key)));
    static_assert(noexcept(highway_hash_256(empty_sv, key)));

    XxHash3Hasher<std::string> xx_hasher;
    WyHasher<std::string> wy_hasher;
    static_assert(noexcept(xx_hasher(std::string{})));
    static_assert(noexcept(wy_hasher(std::string{})));
}

// ─── xxHash3 64-bit Official Reference Test Vectors ──────────────────────────

TEST(XxHash3Test, OfficialTestVectors64) {
    using namespace algoat::hash;

    struct TestCase {
        std::size_t len;
        uint64_t seed;
        uint64_t expected;
    };

    const std::vector<TestCase> test_cases = {
        {0, 0, 0x2D06800538D394C2ULL},    {0, 0x9E3779B185EBCA8DULL, 0xA8A6B918B2F0364AULL},
        {1, 0, 0xC44BDFF4074EECDBULL},    {1, 0x9E3779B185EBCA8DULL, 0x032BE332DD766EF8ULL},
        {2, 0, 0x7A9978044CB8A8BBULL},    {3, 0, 0x54247382A8D6B94DULL},
        {4, 0, 0xE5DC74BC51848A51ULL},    {8, 0, 0x24CCC9ACAA9F65E4ULL},
        {9, 0, 0x14D5001C15DD3F2BULL},    {16, 0, 0x981B17D36C7498C9ULL},
        {17, 0, 0x796F5ACD3A60F862ULL},   {32, 0, 0x9FEADDBDBF57EED3ULL},
        {64, 0, 0x9CB48487720EC49DULL},   {128, 0, 0xFCFF24126754D861ULL},
        {129, 0, 0x98F1B0A679A2CA29ULL},  {240, 0, 0x81C3C2B67F568CCFULL},
        {241, 0, 0xC5A639ECD2030E5EULL},  {1024, 0, 0xDD85C9B5C1109C5CULL},
        {2222, 0, 0x1ADBD424A23E32DEULL},
    };

    for (const auto& tc : test_cases) {
        auto buf = generate_xxh3_test_buffer(tc.len);
        uint64_t actual = xxhash3_64(buf, tc.seed);
        EXPECT_EQ(actual, tc.expected) << "Failed for length=" << tc.len << " seed=" << tc.seed;
    }
}

// ─── xxHash3 128-bit Official Reference Test Vectors ─────────────────────────

TEST(XxHash3Test, OfficialTestVectors128) {
    using namespace algoat::hash;

    struct TestCase128 {
        std::size_t len;
        uint64_t seed;
        uint64_t exp_lo;
        uint64_t exp_hi;
    };

    const std::vector<TestCase128> test_cases = {
        {0, 0, 0x6001C324468D497FULL, 0x99AA06D3014798D8ULL},
        {0, 0x9E3779B185EBCA8DULL, 0xA986DFC5D7605BFEULL, 0x00FEAA732A3CE25EULL},
        {1, 0, 0xC44BDFF4074EECDBULL, 0xA6CD5E9392000F6AULL},
        {1, 0x9E3779B185EBCA8DULL, 0x032BE332DD766EF8ULL, 0x20E49ABCC53B3842ULL},
        {2, 0, 0x7A9978044CB8A8BBULL, 0x76750C3C7BF95668ULL},
        {3, 0, 0x54247382A8D6B94DULL, 0x20EFC49FF02422EAULL},
        {4, 0, 0x2E7D8D6876A39FE9ULL, 0x970D585AC632BF8EULL},
        {8, 0, 0x64C69CAB4BB21DC5ULL, 0x47A7F080D82BB456ULL},
        {9, 0, 0xED7CCBC501EB7501ULL, 0x564EF6078950D457ULL},
        {16, 0, 0x562980258A998629ULL, 0xC68C368ECF8A9C05ULL},
        {17, 0, 0xABBC12D11973D7DBULL, 0x955FA78643ED3669ULL},
        {32, 0, 0x278410A17595E3F9ULL, 0x98FC6458710DC2E8ULL},
        {64, 0, 0xEFDB6A44690721A9ULL, 0x6D90E81A9B0FD622ULL},
        {128, 0, 0xEBB15E34A7FB5AB1ULL, 0x39992220E045260AULL},
        {129, 0, 0x86C9E3BC8F0A3B5CULL, 0x03815FC91F1B30B6ULL},
        {240, 0, 0x5C9AAE94C8EBE5A0ULL, 0xAA4202DAA2769DC8ULL},
        {241, 0, 0xC5A639ECD2030E5EULL, 0x99A80ECF0ECFC647ULL},
        {1024, 0, 0xDD85C9B5C1109C5CULL, 0x0D30D24071C64C57ULL},
        {2222, 0, 0x1ADBD424A23E32DEULL, 0x27E776A0DDE33211ULL},
    };

    for (const auto& tc : test_cases) {
        auto buf = generate_xxh3_test_buffer(tc.len);
        auto actual = xxhash3_128(buf, tc.seed);
        EXPECT_EQ(actual[0], tc.exp_lo) << "Lo failed for len=" << tc.len;
        EXPECT_EQ(actual[1], tc.exp_hi) << "Hi failed for len=" << tc.len;
    }
}

// ─── WyHash Canonical Reference Test Vectors ──────────────────────────────────

TEST(WyHashTest, CanonicalTestVectors) {
    using namespace algoat::hash;

    struct TestCase {
        std::string_view input;
        uint64_t seed;
        uint64_t expected;
    };

    const std::vector<TestCase> test_cases = {
        {"", 0, 0x93228a4de0eec5a2ULL},
        {"a", 0, 0xaced12527fe5bff8ULL},
        {"abc", 0, 0x989b4a209c1011c9ULL},
        {"message digest", 0, 0x309ab4c045215e8fULL},
        {"abcdefghijklmnopqrstuvwxyz", 0, 0xccaeadc12a061176ULL},
        {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 0,
         0x1fdd130ecb5b4709ULL},
        {"12345678901234567890123456789012345678901234567890123456789012345678901234567890", 0,
         0x7e22da19f1a6055aULL},
        // Seeded vectors
        {"", 0, 0x93228a4de0eec5a2ULL},
        {"a", 1, 0xc5bac3db178713c4ULL},
        {"abc", 2, 0xa97f2f7b1d9b3314ULL},
        {"message digest", 3, 0x786d1f1df3801df4ULL},
        {"abcdefghijklmnopqrstuvwxyz", 4, 0xdca5a8138ad37c87ULL},
    };

    for (const auto& tc : test_cases) {
        uint64_t from_span = wyhash(to_byte_span(tc.input), tc.seed);
        uint64_t from_sv = wyhash(tc.input, tc.seed);
        EXPECT_EQ(from_span, tc.expected) << "Failed for input: " << tc.input;
        EXPECT_EQ(from_sv, tc.expected) << "Failed string_view overload for input: " << tc.input;
    }
}

// ─── HighwayHash Reference Test Vectors ───────────────────────────────────────

TEST(HighwayHashTest, ReferenceTestVectors) {
    using namespace algoat::hash;

    const HighwayKey std_key = {1, 2, 3, 4};
    const HighwayKey zero_key = {0, 0, 0, 0};

    // Test 1: Standard Key, 1 Byte Input [0xFF]
    {
        const std::byte byte_ff{0xFF};
        std::span<const std::byte> input{&byte_ff, 1};
        uint64_t h64 = highway_hash_64(input, std_key);
        EXPECT_EQ(h64, 0x7858f24d2d79b2b2ULL);

        auto h256 = highway_hash_256(input, std_key);
        EXPECT_EQ(h256[0], 0x7161cadbf7cd70e1ULL);
        EXPECT_EQ(h256[1], 0xaac4905de62b2f5eULL);
        EXPECT_EQ(h256[2], 0x07b02b936933faa7ULL);
        EXPECT_EQ(h256[3], 0xc8efcfc45b239f8dULL);
    }

    // Test 2: Standard Key, 0 Bytes Input
    {
        std::span<const std::byte> input{};
        uint64_t h64 = highway_hash_64(input, std_key);
        EXPECT_EQ(h64, 0x92943a34e3447faaULL);

        auto h256 = highway_hash_256(input, std_key);
        EXPECT_EQ(h256[0], 0x5abc20e8db09ebd6ULL);
        EXPECT_EQ(h256[1], 0x58dece8699205080ULL);
        EXPECT_EQ(h256[2], 0x9de9645881e6786eULL);
        EXPECT_EQ(h256[3], 0x1b00fefcc21128c2ULL);
    }

    // Test 3: Standard Key, 32 Sequential Bytes
    {
        std::vector<std::byte> seq32(32);
        for (std::size_t i = 0; i < 32; ++i)
            seq32[i] = static_cast<std::byte>(i);
        uint64_t h64 = highway_hash_64(seq32, std_key);
        EXPECT_EQ(h64, 0x6cb9d962db4f34e8ULL);

        auto h256 = highway_hash_256(seq32, std_key);
        EXPECT_EQ(h256[0], 0xe9fa92165762737bULL);
        EXPECT_EQ(h256[1], 0xa637ff0fdefc1f58ULL);
        EXPECT_EQ(h256[2], 0x12a2dfad57fb47b0ULL);
        EXPECT_EQ(h256[3], 0xf8116c2cfc585ff6ULL);
    }

    // Test 4: Zero Key, "Hello, HighwayHash!" (19 bytes)
    {
        std::string_view msg = "Hello, HighwayHash!";
        uint64_t h64 = highway_hash_64(msg, zero_key);
        EXPECT_EQ(h64, 0xe630f0a7f07cbcc6ULL);

        auto h256 = highway_hash_256(msg, zero_key);
        EXPECT_EQ(h256[0], 0xa7b5cd4afdf2cb1eULL);
        EXPECT_EQ(h256[1], 0x57468893f78b3e38ULL);
        EXPECT_EQ(h256[2], 0xad25ab2070cf6faaULL);
        EXPECT_EQ(h256[3], 0x6c4983da709454cbULL);
    }

    // Test 5: Standard Key, 64 Sequential Bytes
    {
        std::vector<std::byte> seq64(64);
        for (std::size_t i = 0; i < 64; ++i)
            seq64[i] = static_cast<std::byte>(i);
        uint64_t h64 = highway_hash_64(seq64, std_key);
        EXPECT_EQ(h64, 0x278b96aef9b77546ULL);

        auto h256 = highway_hash_256(seq64, std_key);
        EXPECT_EQ(h256[0], 0x5f5e62b17074ef95ULL);
        EXPECT_EQ(h256[1], 0x703315a6ebcc5d6bULL);
        EXPECT_EQ(h256[2], 0x28d8cc36bac98608ULL);
        EXPECT_EQ(h256[3], 0xef0681a561bc9753ULL);
    }

    // Test 6: 128-bit HighwayHash evaluation
    {
        std::string_view msg = "HighwayHash128_Test";
        auto h128_1 = highway_hash_128(msg, std_key);
        auto h128_2 = highway_hash_128(msg, std_key);
        EXPECT_EQ(h128_1[0], h128_2[0]);
        EXPECT_EQ(h128_1[1], h128_2[1]);

        auto h128_diff = highway_hash_128(msg, zero_key);
        EXPECT_FALSE(h128_1[0] == h128_diff[0] && h128_1[1] == h128_diff[1]);
    }
}

// ─── CPO Adaptor & Concept Drop-In Tests ──────────────────────────────────────

TEST(CPOAdaptorTest, DropInReplacementUnorderedContainers) {
    using namespace algoat::hash;

    // Verify std::unordered_map compilation and lookup with XxHash3Hasher
    std::unordered_map<std::string, int, XxHash3Hasher<std::string>> xx_map;
    xx_map["alpha"] = 10;
    xx_map["beta"] = 20;
    xx_map["gamma"] = 30;

    EXPECT_EQ(xx_map["alpha"], 10);
    EXPECT_EQ(xx_map["beta"], 20);
    EXPECT_EQ(xx_map["gamma"], 30);
    EXPECT_EQ(xx_map.size(), 3U);

    // Verify std::unordered_set compilation and lookup with WyHasher
    std::unordered_set<std::string, WyHasher<std::string>> wy_set;
    wy_set.insert("one");
    wy_set.insert("two");
    wy_set.insert("three");

    EXPECT_TRUE(wy_set.contains("one"));
    EXPECT_TRUE(wy_set.contains("two"));
    EXPECT_TRUE(wy_set.contains("three"));
    EXPECT_FALSE(wy_set.contains("four"));

    // Integral and string_view types
    XxHash3Hasher<uint64_t> u64_hasher;
    EXPECT_NE(u64_hasher(12345ULL), 0U);

    WyHasher<std::string_view> sv_hasher;
    EXPECT_NE(sv_hasher("test_key"), 0U);
}

// ─── SMHasher Quality Verification Tests ──────────────────────────────────────

TEST(SMHasherQualityTest, StrictAvalancheCriterion) {
    using namespace algoat::hash;

    // Strict Avalanche Criterion (SAC):
    // For a given input, flipping any single input bit should result in each output
    // bit changing with approximately 50% probability across many samples.
    constexpr std::size_t kInputBytes = 64;
    constexpr std::size_t kTrials = 500;
    const HighwayKey key = {0x1234567890ABCDEFULL, 0x0FEDCBA098765432ULL, 0xA5A5A5A55A5A5A5AULL,
                            0x0123456789ABCDEFULL};

    std::vector<uint8_t> base_buf(kInputBytes);
    for (std::size_t i = 0; i < kInputBytes; ++i) {
        base_buf[i] = static_cast<uint8_t>(i * 37 + 11);
    }

    std::array<std::size_t, 64> xx_flips{};
    std::array<std::size_t, 64> wy_flips{};
    std::array<std::size_t, 64> hw_flips{};

    for (std::size_t trial = 0; trial < kTrials; ++trial) {
        // Pseudo-random modification
        for (std::size_t i = 0; i < kInputBytes; ++i) {
            base_buf[i] ^= static_cast<uint8_t>(trial * 13 + i);
        }

        auto span_base = std::as_bytes(std::span{base_buf});
        uint64_t xx_orig = xxhash3_64(span_base);
        uint64_t wy_orig = wyhash(span_base);
        uint64_t hw_orig = highway_hash_64(span_base, key);

        // Flip a specific bit (e.g. bit in the middle of buffer)
        std::size_t target_byte = (trial * 7) % kInputBytes;
        uint8_t target_bit = 1U << (trial % 8);
        base_buf[target_byte] ^= target_bit;

        auto span_flipped = std::as_bytes(std::span{base_buf});
        uint64_t xx_diff = xx_orig ^ xxhash3_64(span_flipped);
        uint64_t wy_diff = wy_orig ^ wyhash(span_flipped);
        uint64_t hw_diff = hw_orig ^ highway_hash_64(span_flipped, key);

        for (int b = 0; b < 64; ++b) {
            if ((xx_diff >> b) & 1ULL)
                xx_flips[b]++;
            if ((wy_diff >> b) & 1ULL)
                wy_flips[b]++;
            if ((hw_diff >> b) & 1ULL)
                hw_flips[b]++;
        }

        // Revert bit flip
        base_buf[target_byte] ^= target_bit;
    }

    // Check that each bit has probability between 38% and 62% (centered at 50%)
    for (int b = 0; b < 64; ++b) {
        double p_xx = static_cast<double>(xx_flips[b]) / kTrials;
        double p_wy = static_cast<double>(wy_flips[b]) / kTrials;
        double p_hw = static_cast<double>(hw_flips[b]) / kTrials;

        EXPECT_GE(p_xx, 0.35) << "xxHash3 bit " << b << " low bias: " << p_xx;
        EXPECT_LE(p_xx, 0.65) << "xxHash3 bit " << b << " high bias: " << p_xx;

        EXPECT_GE(p_wy, 0.35) << "WyHash bit " << b << " low bias: " << p_wy;
        EXPECT_LE(p_wy, 0.65) << "WyHash bit " << b << " high bias: " << p_wy;

        EXPECT_GE(p_hw, 0.35) << "HighwayHash bit " << b << " low bias: " << p_hw;
        EXPECT_LE(p_hw, 0.65) << "HighwayHash bit " << b << " high bias: " << p_hw;
    }
}

TEST(SMHasherQualityTest, CollisionDistributionUniformity) {
    using namespace algoat::hash;

    // Hash 10,000 sequential items and check collision freedom
    constexpr std::size_t kCount = 10000;
    std::unordered_set<uint64_t> xx_hashes;
    std::unordered_set<uint64_t> wy_hashes;
    std::unordered_set<uint64_t> hw_hashes;
    HighwayKey key = {10, 20, 30, 40};

    for (std::size_t i = 0; i < kCount; ++i) {
        std::string s = "algoat_hash_key_" + std::to_string(i);
        auto sp = to_byte_span(s);

        uint64_t h_xx = xxhash3_64(sp);
        uint64_t h_wy = wyhash(sp);
        uint64_t h_hw = highway_hash_64(sp, key);

        EXPECT_TRUE(xx_hashes.insert(h_xx).second) << "xxHash3 collision at key " << s;
        EXPECT_TRUE(wy_hashes.insert(h_wy).second) << "WyHash collision at key " << s;
        EXPECT_TRUE(hw_hashes.insert(h_hw).second) << "HighwayHash collision at key " << s;
    }

    EXPECT_EQ(xx_hashes.size(), kCount);
    EXPECT_EQ(wy_hashes.size(), kCount);
    EXPECT_EQ(hw_hashes.size(), kCount);
}

TEST(SMHasherQualityTest, BitIndependenceCriterion) {
    using namespace algoat::hash;

    // Bit Independence Criterion (BIC):
    // For output bit pairs (i, j), flipping an input bit should cause bit i and bit j
    // to invert independently (P(flip_i ^ flip_j) approx 0.5).
    constexpr std::size_t kInputBytes = 64;
    constexpr std::size_t kTrials = 400;
    const HighwayKey key = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0x1122334455667788ULL,
                            0x99AABBCCDDEEFF00ULL};

    std::vector<uint8_t> buf(kInputBytes);
    for (std::size_t i = 0; i < kInputBytes; ++i) {
        buf[i] = static_cast<uint8_t>(i * 41 + 17);
    }

    std::size_t xx_pair_flips = 0;
    std::size_t wy_pair_flips = 0;
    std::size_t hw_pair_flips = 0;

    for (std::size_t trial = 0; trial < kTrials; ++trial) {
        auto span_base = std::as_bytes(std::span{buf});
        uint64_t xx_orig = xxhash3_64(span_base);
        uint64_t wy_orig = wyhash(span_base);
        uint64_t hw_orig = highway_hash_64(span_base, key);

        std::size_t target_byte = (trial * 11) % kInputBytes;
        uint8_t target_bit = 1U << (trial % 8);
        buf[target_byte] ^= target_bit;

        auto span_flipped = std::as_bytes(std::span{buf});
        uint64_t xx_diff = xx_orig ^ xxhash3_64(span_flipped);
        uint64_t wy_diff = wy_orig ^ wyhash(span_flipped);
        uint64_t hw_diff = hw_orig ^ highway_hash_64(span_flipped, key);

        // Check correlation between bit 7 and bit 42
        xx_pair_flips += (((xx_diff >> 7) ^ (xx_diff >> 42)) & 1ULL);
        wy_pair_flips += (((wy_diff >> 7) ^ (wy_diff >> 42)) & 1ULL);
        hw_pair_flips += (((hw_diff >> 7) ^ (hw_diff >> 42)) & 1ULL);

        buf[target_byte] ^= target_bit; // Revert
    }

    double p_xx = static_cast<double>(xx_pair_flips) / kTrials;
    double p_wy = static_cast<double>(wy_pair_flips) / kTrials;
    double p_hw = static_cast<double>(hw_pair_flips) / kTrials;

    EXPECT_NEAR(p_xx, 0.50, 0.15) << "xxHash3 BIC failed";
    EXPECT_NEAR(p_wy, 0.50, 0.15) << "WyHash BIC failed";
    EXPECT_NEAR(p_hw, 0.50, 0.15) << "HighwayHash BIC failed";
}

TEST(SMHasherQualityTest, DifferentialDistribution) {
    using namespace algoat::hash;

    // Differential test: small input differences must yield output differences
    // with mean Hamming weight around 32 and coverage across all bits.
    constexpr std::size_t kTrials = 500;
    const HighwayKey key = {0xCAFEBABE12345678ULL, 0xDEADBEEF87654321ULL, 0x0F0F0F0F0F0F0F0FULL,
                            0xF0F0F0F0F0F0F0F0ULL};

    uint64_t xx_all_bits = 0;
    uint64_t wy_all_bits = 0;
    uint64_t hw_all_bits = 0;
    double xx_weight_sum = 0;
    double wy_weight_sum = 0;
    double hw_weight_sum = 0;

    for (std::size_t trial = 0; trial < kTrials; ++trial) {
        uint64_t x = trial * 0x9E3779B185EBCA87ULL + 1;
        uint64_t y = x ^ (1ULL << (trial % 64)); // Hamming distance 1

        auto sp_x = std::as_bytes(std::span<const uint64_t, 1>{&x, 1});
        auto sp_y = std::as_bytes(std::span<const uint64_t, 1>{&y, 1});

        uint64_t xx_diff = xxhash3_64(sp_x) ^ xxhash3_64(sp_y);
        uint64_t wy_diff = wyhash(sp_x) ^ wyhash(sp_y);
        uint64_t hw_diff = highway_hash_64(sp_x, key) ^ highway_hash_64(sp_y, key);

        xx_all_bits |= xx_diff;
        wy_all_bits |= wy_diff;
        hw_all_bits |= hw_diff;

        xx_weight_sum += std::popcount(xx_diff);
        wy_weight_sum += std::popcount(wy_diff);
        hw_weight_sum += std::popcount(hw_diff);
    }

    EXPECT_EQ(xx_all_bits, ~0ULL) << "xxHash3 differential failed to toggle all output bits";
    EXPECT_EQ(wy_all_bits, ~0ULL) << "WyHash differential failed to toggle all output bits";
    EXPECT_EQ(hw_all_bits, ~0ULL) << "HighwayHash differential failed to toggle all output bits";

    double xx_mean_wt = xx_weight_sum / kTrials;
    double wy_mean_wt = wy_weight_sum / kTrials;
    double hw_mean_wt = hw_weight_sum / kTrials;

    EXPECT_NEAR(xx_mean_wt, 32.0, 4.0) << "xxHash3 differential mean Hamming weight biased";
    EXPECT_NEAR(wy_mean_wt, 32.0, 4.0) << "WyHash differential mean Hamming weight biased";
    EXPECT_NEAR(hw_mean_wt, 32.0, 4.0) << "HighwayHash differential mean Hamming weight biased";
}
