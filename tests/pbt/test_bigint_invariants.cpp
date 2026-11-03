#include "algoat/numerics/bigint.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <string>
#include <vector>

using namespace algoat::numerics;

namespace rc {

// Generator for BigInt using arbitrary byte vectors
template <> struct Arbitrary<BigInt> {
    static Gen<BigInt> arbitrary() {
        return gen::map(gen::arbitrary<std::vector<uint64_t>>(),
                        [](const std::vector<uint64_t>& limbs) {
                            if (limbs.empty()) {
                                return BigInt();
                            }
                            // Limit to at most 16 limbs (1024 bits) for fast PBT execution
                            size_t count = std::min(limbs.size(), size_t{16});
                            std::string hex_str = "0x";
                            for (size_t i = 0; i < count; ++i) {
                                char buf[32];
                                std::snprintf(buf, sizeof(buf), "%016llx",
                                              static_cast<unsigned long long>(limbs[i]));
                                hex_str += buf;
                            }
                            return BigInt(hex_str, 16);
                        });
    }
};

} // namespace rc

RC_GTEST_PROP(BigIntPBT, AdditionCommutativity, (const BigInt& a, const BigInt& b)) {
    RC_ASSERT(a + b == b + a);
}

RC_GTEST_PROP(BigIntPBT, AdditionAssociativity,
              (const BigInt& a, const BigInt& b, const BigInt& c)) {
    RC_ASSERT((a + b) + c == a + (b + c));
}

RC_GTEST_PROP(BigIntPBT, AdditionIdentityAndInverse, (const BigInt& a)) {
    BigInt zero(0);
    RC_ASSERT(a + zero == a);
    RC_ASSERT(a - a == zero);
    RC_ASSERT(a + (-a) == zero);
}

RC_GTEST_PROP(BigIntPBT, MultiplicationCommutativity, (const BigInt& a, const BigInt& b)) {
    RC_ASSERT(a * b == b * a);
}

RC_GTEST_PROP(BigIntPBT, MultiplicationIdentityAndZero, (const BigInt& a)) {
    BigInt zero(0);
    BigInt one(1);
    RC_ASSERT(a * one == a);
    RC_ASSERT(a * zero == zero);
}

RC_GTEST_PROP(BigIntPBT, Distributivity, (const BigInt& a, const BigInt& b, const BigInt& c)) {
    RC_ASSERT(a * (b + c) == (a * b) + (a * c));
}

RC_GTEST_PROP(BigIntPBT, DivisionTheorem, (const BigInt& a, const BigInt& b_raw)) {
    // Divisor must be non-zero
    BigInt b = b_raw.is_zero() ? BigInt(7) : b_raw;

    BigInt q = a / b;
    BigInt r = a % b;

    RC_ASSERT(q * b + r == a);

    // Magnitude of remainder strictly less than magnitude of divisor
    BigInt abs_r = r.is_negative() ? -r : r;
    BigInt abs_b = b.is_negative() ? -b : b;
    RC_ASSERT(abs_r < abs_b);

    // Remainder sign matches dividend sign (or zero)
    if (!r.is_zero()) {
        RC_ASSERT(r.is_negative() == a.is_negative());
    }
}

RC_GTEST_PROP(BigIntPBT, BitwiseXorInvolution, (const BigInt& a, const BigInt& b)) {
    RC_ASSERT(((a ^ b) ^ b) == a);
}

RC_GTEST_PROP(BigIntPBT, ShiftRoundTrip, (const BigInt& a)) {
    BigInt non_neg = a.is_negative() ? -a : a;
    for (size_t shift : {1, 7, 32, 64, 128}) {
        BigInt shifted = (non_neg << shift) >> shift;
        RC_ASSERT(shifted == non_neg);
    }
}

RC_GTEST_PROP(BigIntPBT, HexStringRoundTrip, (const BigInt& a)) {
    std::string hex = a.to_hex();
    BigInt reconstructed(hex, 16);
    RC_ASSERT(reconstructed == a);
}

RC_GTEST_PROP(BigIntPBT, DecimalStringRoundTrip, (const BigInt& a)) {
    std::string dec = a.to_string();
    BigInt reconstructed(dec);
    RC_ASSERT(reconstructed == a);
}
