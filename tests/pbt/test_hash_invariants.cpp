#include <algoat/hash/hash.hpp>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <span>
#include <string>
#include <vector>

using namespace algoat::hash;

namespace {

std::span<const std::byte> to_span(const std::vector<uint8_t>& v) noexcept {
    return std::as_bytes(std::span{v.data(), v.size()});
}

} // namespace

// ─── 1. Determinism Invariant ─────────────────────────────────────────────────

RC_GTEST_PROP(HashInvariantsPBT, Determinism,
              (const std::vector<uint8_t>& data, uint64_t seed, const HighwayKey& key)) {
    auto span = to_span(data);

    uint64_t xx1 = xxhash3_64(span, seed);
    uint64_t xx2 = xxhash3_64(span, seed);
    RC_ASSERT(xx1 == xx2);

    auto xx128_1 = xxhash3_128(span, seed);
    auto xx128_2 = xxhash3_128(span, seed);
    RC_ASSERT(xx128_1[0] == xx128_2[0]);
    RC_ASSERT(xx128_1[1] == xx128_2[1]);

    uint64_t wy1 = wyhash(span, seed);
    uint64_t wy2 = wyhash(span, seed);
    RC_ASSERT(wy1 == wy2);

    uint64_t hw1 = highway_hash_64(span, key);
    uint64_t hw2 = highway_hash_64(span, key);
    RC_ASSERT(hw1 == hw2);

    auto hw128_1 = highway_hash_128(span, key);
    auto hw128_2 = highway_hash_128(span, key);
    RC_ASSERT(hw128_1 == hw128_2);

    auto hw256_1 = highway_hash_256(span, key);
    auto hw256_2 = highway_hash_256(span, key);
    RC_ASSERT(hw256_1 == hw256_2);
}

// ─── 2. StringView vs Span Equivalence Invariant ──────────────────────────────

RC_GTEST_PROP(HashInvariantsPBT, StringViewSpanEquivalence,
              (const std::string& str, uint64_t seed, const HighwayKey& key)) {
    auto span = std::as_bytes(std::span{str.data(), str.size()});
    std::string_view sv = str;

    RC_ASSERT(xxhash3_64(span, seed) == xxhash3_64(sv, seed));
    RC_ASSERT(xxhash3_128(span, seed) == xxhash3_128(sv, seed));
    RC_ASSERT(wyhash(span, seed) == wyhash(sv, seed));
    RC_ASSERT(highway_hash_64(span, key) == highway_hash_64(sv, key));
    RC_ASSERT(highway_hash_128(span, key) == highway_hash_128(sv, key));
    RC_ASSERT(highway_hash_256(span, key) == highway_hash_256(sv, key));
}

// ─── 3. Seed Sensitivity Invariant ───────────────────────────────────────────

RC_GTEST_PROP(HashInvariantsPBT, SeedSensitivity,
              (const std::vector<uint8_t>& data, uint64_t seed1, uint64_t seed2)) {
    RC_PRE(seed1 != seed2);
    auto span = to_span(data);

    // With 64-bit outputs, hash collision probability is ~ 2^-64
    RC_ASSERT(xxhash3_64(span, seed1) != xxhash3_64(span, seed2));
    RC_ASSERT(wyhash(span, seed1) != wyhash(span, seed2));
}

// ─── 4. Key Sensitivity Invariant (HighwayHash) ───────────────────────────────

RC_GTEST_PROP(HashInvariantsPBT, KeySensitivity,
              (const std::vector<uint8_t>& data, const HighwayKey& key1, const HighwayKey& key2)) {
    RC_PRE(key1 != key2);
    auto span = to_span(data);

    RC_ASSERT(highway_hash_64(span, key1) != highway_hash_64(span, key2));
    RC_ASSERT(highway_hash_128(span, key1) != highway_hash_128(span, key2));
    RC_ASSERT(highway_hash_256(span, key1) != highway_hash_256(span, key2));
}

// ─── 5. Extension Invariant: data vs data + suffix ─────────────────────────────

RC_GTEST_PROP(HashInvariantsPBT, ExtensionSensitivity,
              (const std::vector<uint8_t>& data, uint8_t extra_byte, uint64_t seed,
               const HighwayKey& key)) {
    std::vector<uint8_t> extended = data;
    extended.push_back(extra_byte);

    auto span1 = to_span(data);
    auto span2 = to_span(extended);

    RC_ASSERT(xxhash3_64(span1, seed) != xxhash3_64(span2, seed));
    RC_ASSERT(wyhash(span1, seed) != wyhash(span2, seed));
    RC_ASSERT(highway_hash_64(span1, key) != highway_hash_64(span2, key));
}

// ─── 6. CPO Functor Drop-In Invariants ────────────────────────────────────────

RC_GTEST_PROP(HashInvariantsPBT, CPOFunctorConsistency, (const std::string& str)) {
    XxHash3Hasher<std::string> xx_hasher;
    WyHasher<std::string> wy_hasher;

    std::size_t expected_xx = static_cast<std::size_t>(xxhash3_64(str, 0));
    std::size_t expected_wy = static_cast<std::size_t>(wyhash(str, 0));

    RC_ASSERT(xx_hasher(str) == expected_xx);
    RC_ASSERT(wy_hasher(str) == expected_wy);
}
