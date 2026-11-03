#include "algoat/numerics/bigint.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace algoat::numerics;

namespace {

BigInt parse_limbs(const uint8_t* data, size_t count) {
    if (count == 0)
        return BigInt();
    std::string hex = "0x";
    for (size_t i = 0; i < count; ++i) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%02x", data[i]);
        hex += buf;
    }
    return BigInt(hex, 16);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) {
        return 0;
    }

    size_t half = size / 2;
    BigInt a = parse_limbs(data, half);
    BigInt b = parse_limbs(data + half, size - half);

    // Invariant 1: Addition commutativity & inverse
    BigInt sum1 = a + b;
    BigInt sum2 = b + a;
    assert(sum1 == sum2);
    assert((sum1 - b) == a);
    assert((sum1 - a) == b);

    // Invariant 2: Multiplication commutativity
    BigInt prod1 = a * b;
    BigInt prod2 = b * a;
    assert(prod1 == prod2);

    // Invariant 3: Division theorem
    if (!b.is_zero()) {
        BigInt q = a / b;
        BigInt r = a % b;
        assert((q * b + r) == a);
        BigInt abs_r = r.is_negative() ? -r : r;
        BigInt abs_b = b.is_negative() ? -b : b;
        assert(abs_r < abs_b);
    }

    // Invariant 4: Bitwise involution
    assert(((a ^ b) ^ b) == a);

    // Invariant 5: String round-trip
    std::string hex_a = a.to_hex();
    BigInt recon_a(hex_a, 16);
    assert(recon_a == a);

    return 0;
}
