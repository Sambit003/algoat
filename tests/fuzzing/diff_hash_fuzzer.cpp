#include <algoat/hash/hash.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

using namespace algoat::hash;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) {
        return 0;
    }

    // Extract seed and key deterministically from input if large enough
    uint64_t seed = 0;
    HighwayKey key = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0x1122334455667788ULL,
                      0x99AABBCCDDEEFF00ULL};

    const uint8_t* payload = data;
    size_t payload_len = size;

    if (size >= 8) {
        std::memcpy(&seed, data, sizeof(seed));
        payload += 8;
        payload_len -= 8;
    }

    if (payload_len >= 32) {
        std::memcpy(key.data(), payload, 32);
        payload += 32;
        payload_len -= 32;
    }

    auto span = std::as_bytes(std::span{payload, payload_len});
    std::string_view sv{reinterpret_cast<const char*>(payload), payload_len};

    // 1. xxHash3 differential consistency (span vs string_view)
    uint64_t xx64_span = xxhash3_64(span, seed);
    uint64_t xx64_sv = xxhash3_64(sv, seed);
    assert(xx64_span == xx64_sv);

    auto xx128_span = xxhash3_128(span, seed);
    auto xx128_sv = xxhash3_128(sv, seed);
    assert(xx128_span[0] == xx128_sv[0]);
    assert(xx128_span[1] == xx128_sv[1]);

    // 2. WyHash consistency
    uint64_t wy_span = wyhash(span, seed);
    uint64_t wy_sv = wyhash(sv, seed);
    assert(wy_span == wy_sv);

    // 3. HighwayHash consistency
    uint64_t hw64_span = highway_hash_64(span, key);
    uint64_t hw64_sv = highway_hash_64(sv, key);
    assert(hw64_span == hw64_sv);

    auto hw128_span = highway_hash_128(span, key);
    auto hw128_sv = highway_hash_128(sv, key);
    assert(hw128_span[0] == hw128_sv[0]);
    assert(hw128_span[1] == hw128_sv[1]);

    auto hw256_span = highway_hash_256(span, key);
    auto hw256_sv = highway_hash_256(sv, key);
    assert(hw256_span[0] == hw256_sv[0]);
    assert(hw256_span[1] == hw256_sv[1]);
    assert(hw256_span[2] == hw256_sv[2]);
    assert(hw256_span[3] == hw256_sv[3]);

    // 4. Scalar vs Vectorized Lane Differential check (if exactly 64 bytes)
    if (payload_len >= 64) {
        uint64_t acc_scalar[8] = {0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888};
        uint64_t acc_vector[8] = {0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888};

        detail::xxh3::accumulate_512_scalar(acc_scalar, payload, detail::xxh3::kDefaultSecret);
        detail::xxh3::accumulate_512(acc_vector, payload, detail::xxh3::kDefaultSecret);

        for (int i = 0; i < 8; ++i) {
            assert(acc_scalar[i] == acc_vector[i]);
        }
    }

    // 5. CPO Functor check
    XxHash3Hasher<std::string_view> xx_cpo;
    WyHasher<std::string_view> wy_cpo;
    assert(xx_cpo(sv) == static_cast<size_t>(xxhash3_64(sv, 0)));
    assert(wy_cpo(sv) == static_cast<size_t>(wyhash(sv, 0)));

    return 0;
}
