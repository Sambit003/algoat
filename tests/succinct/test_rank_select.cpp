#include <algoat/succinct/rank_select.hpp>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <vector>

using namespace algoat::succinct;

TEST(RankSelectTest, DefaultConstruction) {
    RankSelectBitVector bv;
    EXPECT_EQ(bv.num_bits(), 0ULL);
    EXPECT_EQ(bv.bit_size(), 0ULL);
    EXPECT_EQ(bv.total_ones(), 0ULL);
    EXPECT_EQ(bv.rank1(0), 0ULL);
    EXPECT_EQ(bv.rank0(0), 0ULL);
    EXPECT_EQ(bv.overhead_bytes(), 0ULL);
}

TEST(RankSelectTest, SmallSpanBoolConstruction) {
    // 0: true, 1: false, 2: true, 3: true, 4: false, 5: false, 6: true
    bool raw_bits[] = {true, false, true, true, false, false, true};
    RankSelectBitVector bv(std::span<const bool>{raw_bits, 7});

    EXPECT_EQ(bv.num_bits(), 7ULL);
    EXPECT_EQ(bv.bit_size(), 7ULL);
    EXPECT_EQ(bv.total_ones(), 4ULL);

    // Test access
    EXPECT_TRUE(bv.access(0));
    EXPECT_FALSE(bv.access(1));
    EXPECT_TRUE(bv.access(2));
    EXPECT_TRUE(bv.access(3));
    EXPECT_FALSE(bv.access(4));
    EXPECT_FALSE(bv.access(5));
    EXPECT_TRUE(bv.access(6));

    // Test rank1 [0, pos)
    EXPECT_EQ(bv.rank1(0), 0ULL);
    EXPECT_EQ(bv.rank1(1), 1ULL);
    EXPECT_EQ(bv.rank1(2), 1ULL);
    EXPECT_EQ(bv.rank1(3), 2ULL);
    EXPECT_EQ(bv.rank1(4), 3ULL);
    EXPECT_EQ(bv.rank1(5), 3ULL);
    EXPECT_EQ(bv.rank1(6), 3ULL);
    EXPECT_EQ(bv.rank1(7), 4ULL);

    // Test rank0 [0, pos)
    for (size_t i = 0; i <= 7; ++i) {
        EXPECT_EQ(bv.rank0(i), i - bv.rank1(i));
    }

    // Test select1 (1-indexed)
    EXPECT_EQ(bv.select1(1), 0ULL);
    EXPECT_EQ(bv.select1(2), 2ULL);
    EXPECT_EQ(bv.select1(3), 3ULL);
    EXPECT_EQ(bv.select1(4), 6ULL);

    // Test select0 (1-indexed)
    EXPECT_EQ(bv.select0(1), 1ULL);
    EXPECT_EQ(bv.select0(2), 4ULL);
    EXPECT_EQ(bv.select0(3), 5ULL);
}

TEST(RankSelectTest, VectorBoolConstruction) {
    std::vector<bool> bits = {true, false, true, true, false, false, true};
    RankSelectBitVector bv(bits);
    EXPECT_EQ(bv.num_bits(), 7ULL);
    EXPECT_EQ(bv.total_ones(), 4ULL);
    EXPECT_TRUE(bv.access(0));
    EXPECT_FALSE(bv.access(1));
    EXPECT_EQ(bv.rank1(7), 4ULL);
    EXPECT_EQ(bv.select1(4), 6ULL);
}

TEST(RankSelectTest, PackedWordsConstruction) {
    // 130 bits across 3 words (64 + 64 + 2)
    std::vector<uint64_t> words = {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0x3ULL};
    RankSelectBitVector bv(std::span<const uint64_t>{words}, 130);

    EXPECT_EQ(bv.num_bits(), 130ULL);
    size_t expected_ones =
        std::popcount(words[0]) + std::popcount(words[1]) + std::popcount(uint64_t(0x3ULL));
    EXPECT_EQ(bv.total_ones(), expected_ones);

    // Verify access
    for (size_t i = 0; i < 130; ++i) {
        bool expected = (words[i / 64] >> (i % 64)) & 1ULL;
        EXPECT_EQ(bv.access(i), expected);
    }

    // Verify rank1 and rank0 at boundaries
    EXPECT_EQ(bv.rank1(0), 0ULL);
    EXPECT_EQ(bv.rank0(0), 0ULL);
    EXPECT_EQ(bv.rank1(130), expected_ones);
    EXPECT_EQ(bv.rank0(130), 130 - expected_ones);
}

TEST(RankSelectTest, AllZerosBitVector) {
    std::vector<bool> bits(1024, false);
    RankSelectBitVector bv(bits);

    EXPECT_EQ(bv.num_bits(), 1024ULL);
    EXPECT_EQ(bv.total_ones(), 0ULL);

    for (size_t i = 0; i <= 1024; ++i) {
        EXPECT_EQ(bv.rank1(i), 0ULL);
        EXPECT_EQ(bv.rank0(i), i);
    }

    for (uint64_t k = 1; k <= 1024; ++k) {
        EXPECT_EQ(bv.select0(k), k - 1);
    }
}

TEST(RankSelectTest, AllOnesBitVector) {
    std::vector<bool> bits(1024, true);
    RankSelectBitVector bv(bits);

    EXPECT_EQ(bv.num_bits(), 1024ULL);
    EXPECT_EQ(bv.total_ones(), 1024ULL);

    for (size_t i = 0; i <= 1024; ++i) {
        EXPECT_EQ(bv.rank1(i), i);
        EXPECT_EQ(bv.rank0(i), 0ULL);
    }

    for (uint64_t k = 1; k <= 1024; ++k) {
        EXPECT_EQ(bv.select1(k), k - 1);
    }
}

TEST(RankSelectTest, NonAlignedLengthsCrossSuperBlock) {
    // Test sizes across 512-bit boundaries: 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049
    std::vector<size_t> sizes = {1,   63,  64,   65,   127,  128,  129,  511,
                                 512, 513, 1023, 1024, 1025, 2047, 2048, 2049};
    std::mt19937_64 rng(42);

    for (size_t n : sizes) {
        std::vector<bool> bits(n);
        uint64_t ones = 0;
        for (size_t i = 0; i < n; ++i) {
            bits[i] = (rng() % 2 == 1);
            if (bits[i])
                ++ones;
        }

        RankSelectBitVector bv(bits);
        EXPECT_EQ(bv.num_bits(), n);
        EXPECT_EQ(bv.total_ones(), ones);

        // Spot check rank
        EXPECT_EQ(bv.rank1(0), 0ULL);
        EXPECT_EQ(bv.rank1(n), ones);

        // Select1 inversion
        for (uint64_t k = 1; k <= ones; ++k) {
            size_t pos = bv.select1(k);
            EXPECT_TRUE(bv.access(pos));
            EXPECT_EQ(bv.rank1(pos), k - 1);
        }

        // Select0 inversion
        uint64_t zeros = n - ones;
        for (uint64_t k = 1; k <= zeros; ++k) {
            size_t pos = bv.select0(k);
            EXPECT_FALSE(bv.access(pos));
            EXPECT_EQ(bv.rank0(pos), k - 1);
        }
    }
}

TEST(RankSelectTest, ComprehensiveRandomBitVector) {
    constexpr size_t N = 100000; // 100k bits across ~195 superblocks
    std::mt19937_64 rng(12345);
    std::vector<bool> bits(N);
    std::vector<size_t> one_positions;
    std::vector<size_t> zero_positions;

    for (size_t i = 0; i < N; ++i) {
        bits[i] = (rng() % 3 == 0); // ~33% density
        if (bits[i]) {
            one_positions.push_back(i);
        } else {
            zero_positions.push_back(i);
        }
    }

    RankSelectBitVector bv(bits);
    EXPECT_EQ(bv.num_bits(), N);
    EXPECT_EQ(bv.total_ones(), one_positions.size());

    // Linear reference scan check at sample positions
    size_t running_ones = 0;
    for (size_t i = 0; i <= N; ++i) {
        if (i % 257 == 0 || i == N) {
            EXPECT_EQ(bv.rank1(i), running_ones) << "Mismatch at pos " << i;
            EXPECT_EQ(bv.rank0(i), i - running_ones) << "Mismatch rank0 at pos " << i;
        }
        if (i < N && bits[i]) {
            ++running_ones;
        }
    }

    // Verify select1 for samples
    for (size_t idx = 0; idx < one_positions.size(); idx += 73) {
        uint64_t k = idx + 1;
        EXPECT_EQ(bv.select1(k), one_positions[idx]);
    }

    // Verify select0 for samples
    for (size_t idx = 0; idx < zero_positions.size(); idx += 73) {
        uint64_t k = idx + 1;
        EXPECT_EQ(bv.select0(k), zero_positions[idx]);
    }
}

TEST(RankSelectTest, OverheadReporting) {
    constexpr size_t N = 1000000; // 1M bits
    std::vector<bool> bits(N, true);
    RankSelectBitVector bv(bits);

    size_t payload_bytes = (N + 7) / 8;
    // Auxiliary select sampling index must be <= 0.4% of bitvector payload
    EXPECT_LE(static_cast<double>(bv.overhead_bytes()), 0.004 * static_cast<double>(payload_bytes));
    EXPECT_GT(bv.overhead_bytes(), 0ULL);
    EXPECT_GT(bv.total_index_bytes(), bv.overhead_bytes());
}

TEST(RankSelectTest, BoundaryAndDeathTests) {
    std::vector<bool> bits = {true, false, true, true, false};
    RankSelectBitVector bv(bits);

    EXPECT_EQ(bv.rank1(0), 0ULL);
    EXPECT_EQ(bv.rank1(5), 3ULL);

    EXPECT_EQ(bv.rank0(0), 0ULL);
    EXPECT_EQ(bv.rank0(5), 2ULL);

    EXPECT_TRUE(bv.access(0));
    EXPECT_FALSE(bv.access(1));
    EXPECT_TRUE(bv.access(3));
    EXPECT_FALSE(bv.access(4));

#if !defined(NDEBUG)
    // In debug builds, out-of-bounds assertions verify API contract
    EXPECT_DEATH((void)bv.rank1(6), "");
    EXPECT_DEATH((void)bv.rank0(6), "");
    EXPECT_DEATH((void)bv.access(5), "");
    EXPECT_DEATH((void)bv.select1(0), "");
    EXPECT_DEATH((void)bv.select1(4), "");
    EXPECT_DEATH((void)bv.select0(0), "");
    EXPECT_DEATH((void)bv.select0(3), "");
#endif
}

TEST(RankSelectTest, BroadwordAndBmi2SelectInWordEquivalence) {
    // 1. Single-bit words (every position 0..63)
    for (size_t bit = 0; bit < 64; ++bit) {
        uint64_t w = (1ULL << bit);
        EXPECT_EQ(detail::select64(w, 0), static_cast<uint32_t>(bit));
        EXPECT_EQ(detail::select64_broadword(w, 0), static_cast<uint32_t>(bit));
    }

    // 2. All-ones word (0..63)
    uint64_t all_ones = ~0ULL;
    for (uint32_t k = 0; k < 64; ++k) {
        EXPECT_EQ(detail::select64(all_ones, k), k);
        EXPECT_EQ(detail::select64_broadword(all_ones, k), k);
    }

    // 3. Alternating words
    uint64_t alt_even = 0x5555555555555555ULL; // bits 0, 2, 4, ...
    for (uint32_t k = 0; k < 32; ++k) {
        EXPECT_EQ(detail::select64(alt_even, k), 2 * k);
        EXPECT_EQ(detail::select64_broadword(alt_even, k), 2 * k);
    }

    uint64_t alt_odd = 0xAAAAAAAAAAAAAAAAULL; // bits 1, 3, 5, ...
    for (uint32_t k = 0; k < 32; ++k) {
        EXPECT_EQ(detail::select64(alt_odd, k), 2 * k + 1);
        EXPECT_EQ(detail::select64_broadword(alt_odd, k), 2 * k + 1);
    }

    // 4. Random words comparison
    std::mt19937_64 rng(999);
    for (size_t trial = 0; trial < 1000; ++trial) {
        uint64_t w = rng();
        uint32_t pop = static_cast<uint32_t>(std::popcount(w));
        for (uint32_t k = 0; k < pop; ++k) {
            uint32_t broadword_res = detail::select64_broadword(w, k);
            uint32_t default_res = detail::select64(w, k);
            EXPECT_EQ(default_res, broadword_res) << "Mismatch for word " << w << " at k=" << k;
        }
    }
}
