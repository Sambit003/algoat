#include <algoat/succinct/elias_fano.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using namespace algoat::succinct;

TEST(EliasFanoTest, EmptySequence) {
    EliasFanoEncoder encoder(0, 100);
    std::vector<uint8_t> buffer = encoder.finalize();

    EliasFanoList list(buffer);
    EXPECT_EQ(list.size(), 0ULL);
    EXPECT_EQ(list.universe(), 100ULL);
    EXPECT_EQ(list.begin(), list.end());
    EXPECT_EQ(list.lower_bound_pos(50), 0ULL);
    EXPECT_EQ(list.next_geq(50), 100ULL);
}

TEST(EliasFanoTest, VignaCanonicalExample) {
    // Vigna (2013) Section 4: [5, 8, 8, 15, 32], universe = 36
    const std::vector<uint64_t> expected = {5, 8, 8, 15, 32};
    const uint64_t universe = 36;

    EliasFanoEncoder encoder(expected.size(), universe);
    for (uint64_t val : expected) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();

    // Test owning vector constructor
    EliasFanoList list(buffer);
    EXPECT_EQ(list.size(), 5ULL);
    EXPECT_EQ(list.universe(), 36ULL);

    // Random access at(i)
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(list.at(i), expected[i]);
    }

    // Successor search next_geq and lower_bound_pos
    EXPECT_EQ(list.lower_bound_pos(0), 0ULL);
    EXPECT_EQ(list.next_geq(0), 5ULL);

    EXPECT_EQ(list.lower_bound_pos(5), 0ULL);
    EXPECT_EQ(list.next_geq(5), 5ULL);

    EXPECT_EQ(list.lower_bound_pos(6), 1ULL);
    EXPECT_EQ(list.next_geq(6), 8ULL);

    EXPECT_EQ(list.lower_bound_pos(8), 1ULL);
    EXPECT_EQ(list.next_geq(8), 8ULL);

    EXPECT_EQ(list.lower_bound_pos(9), 3ULL);
    EXPECT_EQ(list.next_geq(9), 15ULL);

    EXPECT_EQ(list.lower_bound_pos(15), 3ULL);
    EXPECT_EQ(list.next_geq(15), 15ULL);

    // Vigna paper trace: skip to 22 -> should land on index 4 (value 32)
    EXPECT_EQ(list.lower_bound_pos(22), 4ULL);
    EXPECT_EQ(list.next_geq(22), 32ULL);

    EXPECT_EQ(list.lower_bound_pos(32), 4ULL);
    EXPECT_EQ(list.next_geq(32), 32ULL);

    // Out of bounds / past maximum
    EXPECT_EQ(list.lower_bound_pos(33), 5ULL);
    EXPECT_EQ(list.next_geq(33), universe);

    EXPECT_EQ(list.lower_bound_pos(50), 5ULL);
    EXPECT_EQ(list.next_geq(50), universe);
}

TEST(EliasFanoTest, ZeroCopySpanBorrow) {
    const std::vector<uint64_t> expected = {3, 10, 100, 450, 1000};
    const uint64_t universe = 2000;

    EliasFanoEncoder encoder(expected.size(), universe);
    for (uint64_t val : expected) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();

    // Construct borrowing from std::span<const uint8_t>
    EliasFanoList list(std::span<const uint8_t>{buffer.data(), buffer.size()});
    EXPECT_EQ(list.size(), expected.size());
    EXPECT_EQ(list.universe(), universe);

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(list.at(i), expected[i]);
    }
}

TEST(EliasFanoTest, ConstIteratorTraversal) {
    const std::vector<uint64_t> expected = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
    const uint64_t universe = 150;

    EliasFanoEncoder encoder(expected.size(), universe);
    for (uint64_t val : expected) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    // Test iteration via range-based for
    std::vector<uint64_t> collected;
    for (uint64_t val : list) {
        collected.push_back(val);
    }
    EXPECT_EQ(collected, expected);

    // Test iterator properties
    auto it = list.begin();
    EXPECT_EQ(*it, 1ULL);
    EXPECT_EQ(*(++it), 4ULL);
    EXPECT_EQ(*(it++), 4ULL);
    EXPECT_EQ(*it, 9ULL);

    size_t count = 0;
    for (auto iter = list.begin(); iter != list.end(); ++iter) {
        EXPECT_EQ(*iter, expected[count++]);
    }
    EXPECT_EQ(count, expected.size());
}

TEST(EliasFanoTest, DenseSequenceZeroLowerBits) {
    // When u <= n, l = 0 (pure unary upper bits)
    const std::vector<uint64_t> expected = {0, 1, 2, 2, 3, 4, 5};
    const uint64_t universe = 6;

    EliasFanoEncoder encoder(expected.size(), universe);
    for (uint64_t val : expected) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    EXPECT_EQ(list.size(), expected.size());
    EXPECT_EQ(list.universe(), universe);

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(list.at(i), expected[i]);
    }

    EXPECT_EQ(list.lower_bound_pos(2), 2ULL);
    EXPECT_EQ(list.next_geq(2), 2ULL);
    EXPECT_EQ(list.lower_bound_pos(6), expected.size());
    EXPECT_EQ(list.next_geq(6), universe);
}

TEST(EliasFanoTest, AllIdenticalElements) {
    const std::vector<uint64_t> expected = {42, 42, 42, 42, 42};
    const uint64_t universe = 100;

    EliasFanoEncoder encoder(expected.size(), universe);
    for (uint64_t val : expected) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(list.at(i), 42ULL);
    }

    EXPECT_EQ(list.lower_bound_pos(10), 0ULL);
    EXPECT_EQ(list.next_geq(10), 42ULL);

    EXPECT_EQ(list.lower_bound_pos(42), 0ULL);
    EXPECT_EQ(list.next_geq(42), 42ULL);

    EXPECT_EQ(list.lower_bound_pos(43), 5ULL);
    EXPECT_EQ(list.next_geq(43), universe);
}

TEST(EliasFanoTest, SingleElement) {
    EliasFanoEncoder encoder(1, 1000);
    encoder.push(500);
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    EXPECT_EQ(list.size(), 1ULL);
    EXPECT_EQ(list.at(0), 500ULL);
    EXPECT_EQ(list.lower_bound_pos(100), 0ULL);
    EXPECT_EQ(list.next_geq(100), 500ULL);
    EXPECT_EQ(list.lower_bound_pos(500), 0ULL);
    EXPECT_EQ(list.next_geq(500), 500ULL);
    EXPECT_EQ(list.lower_bound_pos(501), 1ULL);
    EXPECT_EQ(list.next_geq(501), 1000ULL);
}

TEST(EliasFanoTest, BoundaryUniverseElements) {
    const std::vector<uint64_t> expected = {0, 999};
    const uint64_t universe = 1000;

    EliasFanoEncoder encoder(expected.size(), universe);
    for (uint64_t val : expected) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    EXPECT_EQ(list.at(0), 0ULL);
    EXPECT_EQ(list.at(1), 999ULL);
    EXPECT_EQ(list.lower_bound_pos(999), 1ULL);
    EXPECT_EQ(list.next_geq(999), 999ULL);
    EXPECT_EQ(list.lower_bound_pos(1000), 2ULL);
    EXPECT_EQ(list.next_geq(1000), universe);
}

TEST(EliasFanoTest, InvalidHeaderThrows) {
    std::vector<uint8_t> tiny_buf(16, 0);
    std::span<const uint8_t> tiny_span{tiny_buf};
    EXPECT_THROW((void)EliasFanoList{tiny_span}, std::invalid_argument);

    std::vector<uint8_t> corrupted_buf(64, 0);
    std::span<const uint8_t> corrupted_span{corrupted_buf};
    EXPECT_THROW((void)EliasFanoList{corrupted_span}, std::invalid_argument);
}

TEST(EliasFanoTest, LargeUniformSequenceRoundTrip) {
    const size_t n = 1000000;
    const uint64_t universe = 50000000;

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<uint64_t> dist(0, universe - 1);

    std::vector<uint64_t> seq(n);
    for (size_t i = 0; i < n; ++i) {
        seq[i] = dist(rng);
    }
    std::sort(seq.begin(), seq.end());

    EliasFanoEncoder encoder(n, universe);
    for (uint64_t val : seq) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    EXPECT_EQ(list.size(), n);
    EXPECT_EQ(list.universe(), universe);

    // Verify at(i) samples
    for (size_t i = 0; i < n; i += 1000) {
        EXPECT_EQ(list.at(i), seq[i]);
    }

    // Verify iterator full round-trip
    size_t idx = 0;
    for (uint64_t val : list) {
        ASSERT_EQ(val, seq[idx++]);
    }
    EXPECT_EQ(idx, n);

    // Verify next_geq against std::lower_bound
    for (size_t q = 0; q < 500; ++q) {
        uint64_t query_val = dist(rng);
        auto it = std::lower_bound(seq.begin(), seq.end(), query_val);
        size_t expected_pos = static_cast<size_t>(std::distance(seq.begin(), it));
        uint64_t expected_val = (it != seq.end()) ? *it : universe;

        EXPECT_EQ(list.lower_bound_pos(query_val), expected_pos);
        EXPECT_EQ(list.next_geq(query_val), expected_val);
    }
}

TEST(EliasFanoTest, ZipfianDistributionRoundTrip) {
    const size_t n = 1000000;
    const uint64_t universe = 50000000;

    std::mt19937_64 rng(42);
    // Skewed Zipfian-like distribution using power of uniform
    std::uniform_real_distribution<double> udist(0.0, 1.0);

    std::vector<uint64_t> seq(n);
    for (size_t i = 0; i < n; ++i) {
        double r = udist(rng);
        // Transform to Zipf-like cluster around low values with long tail
        uint64_t val = static_cast<uint64_t>(std::pow(r, 3.0) * (universe - 1));
        seq[i] = std::min(val, universe - 1);
    }
    std::sort(seq.begin(), seq.end());

    EliasFanoEncoder encoder(n, universe);
    for (uint64_t val : seq) {
        encoder.push(val);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    for (size_t i = 0; i < n; i += 1000) {
        EXPECT_EQ(list.at(i), seq[i]);
    }

    // Verify iterator
    size_t idx = 0;
    for (uint64_t val : list) {
        ASSERT_EQ(val, seq[idx++]);
    }
    EXPECT_EQ(idx, n);

    // Query validation
    for (size_t q = 0; q < 200; ++q) {
        uint64_t query_val = static_cast<uint64_t>(udist(rng) * universe);
        auto it = std::lower_bound(seq.begin(), seq.end(), query_val);
        size_t expected_pos = static_cast<size_t>(std::distance(seq.begin(), it));
        uint64_t expected_val = (it != seq.end()) ? *it : universe;

        EXPECT_EQ(list.lower_bound_pos(query_val), expected_pos);
        EXPECT_EQ(list.next_geq(query_val), expected_val);
    }
}

TEST(EliasFanoTest, SpaceEfficiencyWithinTheoreticalBound) {
    const size_t n = 1000000;
    const uint64_t universe = 1000000000; // u = 10^9

    EliasFanoEncoder encoder(n, universe);
    uint64_t step = universe / (n + 1);
    for (size_t i = 0; i < n; ++i) {
        encoder.push((i + 1) * step);
    }
    std::vector<uint8_t> buffer = encoder.finalize();
    EliasFanoList list(buffer);

    // Calculate theoretical space: n * (ceil(log2(u/n)) + 2) bits
    size_t l = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(universe) / n)));
    double theoretical_bytes = (static_cast<double>(n) * (l + 2)) / 8.0;

    size_t actual_bytes = list.memory_usage_bytes();
    // Check that actual memory usage is within 10% of theoretical
    EXPECT_LE(static_cast<double>(actual_bytes), theoretical_bytes * 1.10);
}
