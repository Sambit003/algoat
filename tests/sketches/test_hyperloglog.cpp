#include "algoat/sketches/detail/hash.hpp"
#include "algoat/sketches/hyperloglog.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// 1. Static Contract & Compile-Time Constraints
// ============================================================================

TEST(HyperLogLogTest, StaticPropertiesMatchPrecision) {
    EXPECT_EQ(HyperLogLog<4>::precision, 4);
    EXPECT_EQ(HyperLogLog<4>::register_count, 16);

    EXPECT_EQ(HyperLogLog<14>::precision, 14);
    EXPECT_EQ(HyperLogLog<14>::register_count, 16384);

    EXPECT_EQ(HyperLogLog<18>::precision, 18);
    EXPECT_EQ(HyperLogLog<18>::register_count, 262144);

    // Expected relative error formula: 1.04 / sqrt(m)
    constexpr double expected_err_14 = 1.04 / 128.0;
    EXPECT_NEAR(HyperLogLog<14>::expected_relative_error, expected_err_14, 1e-9);
}

// ============================================================================
// 2. Initial State & Empty Sketch
// ============================================================================

TEST(HyperLogLogTest, EmptySketchHasZeroCount) {
    HyperLogLog<14> hll;
    EXPECT_EQ(hll.count(), 0ULL);
    EXPECT_DOUBLE_EQ(hll.estimate(), 0.0);
    EXPECT_TRUE(hll.is_sparse());
    EXPECT_EQ(hll.mode(), HyperLogLog<14>::Mode::Sparse);
    EXPECT_GT(hll.memory_usage_bytes(), 0);
}

// ============================================================================
// 3. Repeated Additions & Idempotence
// ============================================================================

TEST(HyperLogLogTest, RepeatedItemDoesNotIncreaseCardinality) {
    HyperLogLog<14> hll;
    for (int i = 0; i < 100; ++i) {
        hll.add(std::string("constant_key"));
    }
    EXPECT_EQ(hll.count(), 1ULL);
}

TEST(HyperLogLogTest, DistinctItemsIncreaseCardinality) {
    HyperLogLog<14> hll;
    for (int i = 0; i < 50; ++i) {
        hll.add(i);
    }
    EXPECT_GE(hll.count(), 45ULL);
    EXPECT_LE(hll.count(), 55ULL);
}

// ============================================================================
// 4. Heterogeneous Item Hashing Support
// ============================================================================

struct CustomType {
    int id;
    std::string tag;
    bool operator==(const CustomType& other) const = default;
};

namespace std {
template <> struct hash<CustomType> {
    std::size_t operator()(const CustomType& c) const noexcept {
        return std::hash<int>{}(c.id) ^ (std::hash<std::string>{}(c.tag) << 1);
    }
};
} // namespace std

TEST(HyperLogLogTest, SupportsHeterogeneousDataTypes) {
    HyperLogLog<14> hll;
    // Integers
    hll.add(42);
    hll.add(-100);
    // Floats
    hll.add(3.14159f);
    hll.add(2.718281828459045);
    // Pointers
    const char* ptr = "algoat";
    hll.add(ptr);
    // Strings & StringViews
    hll.add(std::string("hello"));
    hll.add(std::string_view("world"));
    // Custom user struct
    hll.add(CustomType{1, "alpha"});
    hll.add(CustomType{2, "beta"});

    EXPECT_GE(hll.count(), 8ULL);
    EXPECT_LE(hll.count(), 10ULL);
}

// ============================================================================
// 5. Accuracy & Standard Error Bounded Across Scales & Precisions
// ============================================================================

TEST(HyperLogLogTest, AccuracyWithinTheoreticalBoundsP14) {
    constexpr std::size_t P = 14;
    constexpr std::size_t m = 1ULL << P;
    const double theoretical_std_err = 1.04 / std::sqrt(static_cast<double>(m));
    const double max_allowed_rel_err = theoretical_std_err * 2.5; // Within 2.5 sigma

    const std::vector<std::size_t> test_cardinalities = {100, 500, 2000, 10000, 50000, 100000};

    for (std::size_t n : test_cardinalities) {
        HyperLogLog<P> hll;
        for (std::size_t i = 0; i < n; ++i) {
            hll.add_hash(detail::splitmix64(i + 1));
        }

        double est = hll.estimate();
        double rel_err = std::abs(est - static_cast<double>(n)) / static_cast<double>(n);
        EXPECT_LT(rel_err, max_allowed_rel_err)
            << "Cardinality " << n << " estimate " << est << " exceeded error threshold "
            << max_allowed_rel_err;
    }
}

TEST(HyperLogLogTest, AccuracyAcrossMultiplePrecisions) {
    // Test P = 4, 8, 12, 16
    auto test_p = []<std::size_t P>(std::size_t n) {
        constexpr std::size_t m = 1ULL << P;
        const double rse = 1.04 / std::sqrt(static_cast<double>(m));
        HyperLogLog<P> hll;
        for (std::size_t i = 0; i < n; ++i) {
            hll.add_hash(detail::splitmix64(i + 12345));
        }
        double est = hll.estimate();
        double rel_err = std::abs(est - static_cast<double>(n)) / static_cast<double>(n);
        EXPECT_LT(rel_err, rse * 3.0);
    };

    test_p.operator()<4>(20);
    test_p.operator()<8>(200);
    test_p.operator()<12>(2000);
    test_p.operator()<16>(20000);
}

// ============================================================================
// 6. Linear Counting & Threshold Smoothness (Zero Discontinuity)
// ============================================================================

TEST(HyperLogLogTest, ContinuousMonotonicEstimateAcrossThreshold) {
    constexpr std::size_t P = 14;
    HyperLogLog<P> hll;

    // Linear counting threshold for P=14 is 50000
    // Check points around threshold: 45000 to 55000
    double prev_est = 0.0;
    for (std::size_t i = 1; i <= 60000; ++i) {
        hll.add_hash(detail::splitmix64(i));
        if (i % 5000 == 0) {
            double est = hll.estimate();
            EXPECT_GT(est, prev_est) << "Estimate must strictly increase as items are added";
            prev_est = est;
        }
    }
}

// ============================================================================
// 7. Sparse-to-Dense Promotion
// ============================================================================

TEST(HyperLogLogTest, SparseToDensePromotion) {
    HyperLogLog<10> hll; // m = 1024, threshold = 6 * 1024 / 8 = 768 bytes
    EXPECT_TRUE(hll.is_sparse());

    // Add elements until promotion occurs
    for (int i = 0; i < 2000; ++i) {
        hll.add(i);
    }

    EXPECT_FALSE(hll.is_sparse());
    EXPECT_EQ(hll.mode(), HyperLogLog<10>::Mode::Dense);
    EXPECT_GT(hll.count(), 1800ULL);
}

// ============================================================================
// 8. Merging Invariants
// ============================================================================

TEST(HyperLogLogTest, MergeInvariants) {
    HyperLogLog<14> a;
    HyperLogLog<14> b;

    for (int i = 0; i < 1000; ++i) {
        a.add(i);
    }
    for (int i = 500; i < 1500; ++i) {
        b.add(i);
    }

    HyperLogLog<14> merged = a;
    merged.merge(b);

    // Total distinct: 1500
    EXPECT_NEAR(static_cast<double>(merged.count()), 1500.0, 1500.0 * 0.05);

    // Idempotent: A.merge(A) == A
    HyperLogLog<14> a_dup = a;
    a_dup.merge(a);
    EXPECT_EQ(a_dup.count(), a.count());

    // Commutative: A.merge(B) == B.merge(A)
    HyperLogLog<14> b_merged = b;
    b_merged.merge(a);
    EXPECT_EQ(merged.count(), b_merged.count());
}

TEST(HyperLogLogTest, MergeHeterogeneousModes) {
    // A is sparse, B is dense
    HyperLogLog<10> a_sparse;
    for (int i = 0; i < 100; ++i) {
        a_sparse.add(i);
    }
    EXPECT_TRUE(a_sparse.is_sparse());

    HyperLogLog<10> b_dense;
    for (int i = 50; i < 2000; ++i) {
        b_dense.add(i);
    }
    EXPECT_FALSE(b_dense.is_sparse());

    // Sparse merges Dense
    HyperLogLog<10> m1 = a_sparse;
    m1.merge(b_dense);
    EXPECT_FALSE(m1.is_sparse());
    EXPECT_NEAR(static_cast<double>(m1.count()), 2000.0, 2000.0 * 0.1);

    // Dense merges Sparse
    HyperLogLog<10> m2 = b_dense;
    m2.merge(a_sparse);
    EXPECT_FALSE(m2.is_sparse());
    EXPECT_NEAR(static_cast<double>(m2.count()), 2000.0, 2000.0 * 0.1);
}

// ============================================================================
// 9. Redis Wire Format Compatibility (P = 14)
// ============================================================================

TEST(HyperLogLogTest, RedisWireFormatRoundTripP14) {
    HyperLogLog<14> hll;
    for (int i = 0; i < 5000; ++i) {
        hll.add(i);
    }

    auto bytes = hll.serialize();
    // Redis dense wire format has fixed size: 16-byte header + 12288 bytes = 12304 bytes
    EXPECT_EQ(bytes.size(), 12304);

    // Check magic bytes "HYLL"
    EXPECT_EQ(bytes[0], 'H');
    EXPECT_EQ(bytes[1], 'Y');
    EXPECT_EQ(bytes[2], 'L');
    EXPECT_EQ(bytes[3], 'L');

    // Encoding: 0 = HLL_DENSE
    EXPECT_EQ(bytes[4], 0);

    // Redis reserved notused[3] bytes must strictly be zero
    EXPECT_EQ(bytes[5], 0);
    EXPECT_EQ(bytes[6], 0);
    EXPECT_EQ(bytes[7], 0);

    // Deserialization
    HyperLogLog<14> restored = HyperLogLog<14>::deserialize(bytes);
    EXPECT_EQ(restored.count(), hll.count());
    EXPECT_EQ(restored.mode(), HyperLogLog<14>::Mode::Dense);
}

TEST(HyperLogLogTest, SerializeSparsePreservesConstInstanceSparseMode) {
    const HyperLogLog<14> hll;
    auto bytes = hll.serialize();
    EXPECT_EQ(bytes.size(), 12304);
    EXPECT_TRUE(hll.is_sparse());
}

TEST(HyperLogLogTest, SparseDeduplicationPreservesExactLinearCounting) {
    HyperLogLog<14> hll;
    for (int i = 0; i < 200; ++i) {
        hll.add(i);
        hll.add(i); // duplicates
    }
    EXPECT_TRUE(hll.is_sparse());
    EXPECT_NEAR(static_cast<double>(hll.count()), 200.0, 5.0);
}

TEST(HyperLogLogTest, EmpiricalStandardErrorAcrossTrials) {
    constexpr std::size_t P = 12;
    constexpr std::size_t m = 1ULL << P;
    const double theoretical_rse = 1.04 / std::sqrt(static_cast<double>(m));
    constexpr std::size_t num_trials = 30;
    constexpr std::size_t cardinality = 20000;

    double sum_squared_rel_error = 0.0;
    for (std::size_t trial = 0; trial < num_trials; ++trial) {
        HyperLogLog<P> hll;
        const uint64_t seed = 0x9e3779b97f4a7c15ULL * (trial + 1);
        for (std::size_t i = 0; i < cardinality; ++i) {
            hll.add_hash(detail::splitmix64(seed + i));
        }
        double est = hll.estimate();
        double rel_error =
            (est - static_cast<double>(cardinality)) / static_cast<double>(cardinality);
        sum_squared_rel_error += rel_error * rel_error;
    }

    double empirical_rse = std::sqrt(sum_squared_rel_error / static_cast<double>(num_trials));
    EXPECT_LE(empirical_rse, theoretical_rse * 1.15)
        << "Empirical RSE " << empirical_rse
        << " exceeded bound for P=12: " << theoretical_rse * 1.15;
}

TEST(HyperLogLogTest, LargeCardinalityOneMillionP14) {
    constexpr std::size_t P = 14;
    constexpr std::size_t m = 1ULL << P;
    const double theoretical_rse = 1.04 / std::sqrt(static_cast<double>(m));
    constexpr std::size_t N = 1000000;

    HyperLogLog<P> hll;
    for (std::size_t i = 0; i < N; ++i) {
        hll.add_hash(detail::splitmix64(i + 0xDEADBEEF));
    }

    double est = hll.estimate();
    double rel_error = std::abs(est - static_cast<double>(N)) / static_cast<double>(N);
    EXPECT_LT(rel_error, theoretical_rse * 2.5);
}

TEST(HyperLogLogTest, DeserializationRejectsCorruptedHeader) {
    std::vector<uint8_t> corrupted(12304, 0);
    // Invalid magic
    corrupted[0] = 'B';
    corrupted[1] = 'A';
    corrupted[2] = 'D';
    corrupted[3] = '!';

    EXPECT_THROW(HyperLogLog<14>::deserialize(corrupted), std::invalid_argument);

    // Too small buffer
    std::vector<uint8_t> small_buf(10, 0);
    EXPECT_THROW(HyperLogLog<14>::deserialize(small_buf), std::invalid_argument);
}

TEST(HyperLogLogTest, DeserializationParsesRedisSparseRLE) {
    // Construct a minimal valid Redis sparse payload:
    // Header (16 bytes) + XZERO of 16384 zeros (2 bytes) = 18 bytes
    std::vector<uint8_t> sparse_wire(18, 0);
    sparse_wire[0] = 'H';
    sparse_wire[1] = 'Y';
    sparse_wire[2] = 'L';
    sparse_wire[3] = 'L';
    sparse_wire[4] = 1; // HLL_SPARSE
    // XZERO for 16384: count - 1 = 16383 = 0x3FFF
    // Opcode: 01xxxxxx yyyyyyyy -> b0 = 0x40 | 0x3F = 0x7F, b1 = 0xFF
    sparse_wire[16] = 0x7F;
    sparse_wire[17] = 0xFF;

    HyperLogLog<14> restored = HyperLogLog<14>::deserialize(sparse_wire);
    EXPECT_EQ(restored.count(), 0ULL);
}

// ============================================================================
// 10. Clear & Reset
// ============================================================================

TEST(HyperLogLogTest, ClearResetsToEmptySparseState) {
    HyperLogLog<14> hll;
    for (int i = 0; i < 1000; ++i) {
        hll.add(i);
    }
    EXPECT_GT(hll.count(), 0ULL);

    hll.clear();
    EXPECT_EQ(hll.count(), 0ULL);
    EXPECT_TRUE(hll.is_sparse());
}
