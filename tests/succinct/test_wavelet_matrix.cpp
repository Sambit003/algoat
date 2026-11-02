#include <algoat/succinct/wavelet_matrix.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using namespace algoat::succinct;

// ============================================================================
// Acceptance Criteria Verification
// ============================================================================

TEST(WaveletMatrixTest, MemoryEfficiencyLargeSequence) {
    // Acceptance criterion 3:
    // Total memory usage for n = 10^7 uint8_t sequence is <= 1.5x the raw sequence size
    // (raw = 10 MB, index <= 15 MB).
    const size_t n = 10'000'000;
    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<uint8_t>(i % 256);
    }

    WaveletMatrix<uint8_t> wm(data);
    EXPECT_EQ(wm.size(), n);

    size_t mem_bytes = wm.memory_usage_bytes();
    size_t raw_bytes = n * sizeof(uint8_t);
    // Index <= 1.5x raw sequence size:
    EXPECT_LE(mem_bytes, static_cast<size_t>(1.5 * raw_bytes))
        << "Memory bytes: " << mem_bytes << " vs limit: " << static_cast<size_t>(1.5 * raw_bytes);
}

TEST(WaveletMatrixTest, FunctionalCorrectnessLargeSequenceStress) {
    // Acceptance criterion 1:
    // access(i), rank(c, i), select(c, k) produce results identical to brute-force
    // on 10^5 random queries on sequence of length 10^6.
    const size_t n = 1'000'000;
    std::vector<uint32_t> data(n);
    std::mt19937_64 rng(54321);
    std::uniform_int_distribution<uint32_t> dist(0, 1000);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    WaveletMatrix<uint32_t> wm(data);
    EXPECT_EQ(wm.size(), n);

    // Compute ground-truth occurrences per character
    std::vector<std::vector<size_t>> positions_by_char(1001);
    for (size_t i = 0; i < n; ++i) {
        positions_by_char[data[i]].push_back(i);
    }

    // Run 100,000 random queries across access, rank, select
    std::uniform_int_distribution<size_t> pos_dist(0, n - 1);
    std::uniform_int_distribution<uint32_t> char_dist(0, 1000);

    for (size_t q = 0; q < 100'000; ++q) {
        // Test access
        size_t idx = pos_dist(rng);
        EXPECT_EQ(wm.access(idx), data[idx]);

        // Test rank
        uint32_t c = char_dist(rng);
        size_t pref = pos_dist(rng);
        const auto& pos_vec = positions_by_char[c];
        size_t expected_rank = static_cast<size_t>(
            std::lower_bound(pos_vec.begin(), pos_vec.end(), pref) - pos_vec.begin());
        EXPECT_EQ(wm.rank(c, pref), expected_rank);

        // Test select
        if (!pos_vec.empty()) {
            std::uniform_int_distribution<size_t> k_dist(1, pos_vec.size());
            size_t k = k_dist(rng);
            size_t expected_sel = pos_vec[k - 1];
            EXPECT_EQ(wm.select(c, k), expected_sel);
        }
    }
}

TEST(WaveletMatrixTest, RangeCountPaperExample) {
    // Claude & Navarro (2015) Fig. 4 example:
    // A = [4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7]
    const std::vector<uint8_t> data = {4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7};
    WaveletMatrix<uint8_t> wm(data);

    // Full range [0, 13)
    EXPECT_EQ(wm.range_count(0, 13, 0, 7), 13ULL);
    EXPECT_EQ(wm.range_count(0, 13, 1, 3), 6ULL);  // 1 (3), 2 (2), 3 (1)
    EXPECT_EQ(wm.range_count(0, 13, 4, 6), 4ULL);  // 4 (2), 5 (1), 6 (1)
    EXPECT_EQ(wm.range_count(0, 13, 8, 20), 0ULL); // none

    // Boundary conditions
    EXPECT_EQ(wm.range_count(0, 13, 5, 2), 0ULL); // val_lo > val_hi
    EXPECT_EQ(wm.range_count(5, 5, 0, 10), 0ULL); // l == r
    EXPECT_EQ(wm.range_count(7, 3, 0, 10), 0ULL); // l > r

    // Subrange [4, 11) -> [3, 2, 1, 0, 2, 1, 4]
    EXPECT_EQ(wm.range_count(4, 11, 1, 2), 4ULL); // 1 (2), 2 (2)
    EXPECT_EQ(wm.range_count(4, 11, 5, 7), 0ULL);
    EXPECT_EQ(wm.range_count(4, 11, 0, 4), 7ULL); // all in subrange

    // Single value range_count should match rank difference
    for (uint8_t c = 0; c <= 7; ++c) {
        EXPECT_EQ(wm.range_count(4, 11, c, c), wm.rank(c, 11) - wm.rank(c, 4));
    }
}

TEST(WaveletMatrixTest, RangeCountRandomSubranges) {
    std::mt19937_64 rng(31415);
    const size_t n = 200;
    std::vector<uint16_t> data(n);
    std::uniform_int_distribution<uint16_t> val_dist(0, 100);
    for (size_t i = 0; i < n; ++i) {
        data[i] = val_dist(rng);
    }

    WaveletMatrix<uint16_t> wm(data);

    std::uniform_int_distribution<size_t> pos_dist(0, n);
    for (size_t trial = 0; trial < 100; ++trial) {
        size_t l = pos_dist(rng);
        size_t r = pos_dist(rng);
        if (l > r)
            std::swap(l, r);

        uint16_t val1 = val_dist(rng);
        uint16_t val2 = val_dist(rng);
        uint16_t lo = std::min(val1, val2);
        uint16_t hi = std::max(val1, val2);

        size_t expected = 0;
        for (size_t i = l; i < r; ++i) {
            if (data[i] >= lo && data[i] <= hi) {
                ++expected;
            }
        }

        EXPECT_EQ(wm.range_count(l, r, lo, hi), expected)
            << "Mismatch for range [" << l << ", " << r << ") in values [" << lo << ", " << hi
            << "]";
    }
}

TEST(WaveletMatrixTest, RangeQuantilePaperExample) {
    // Claude & Navarro (2015) Fig. 4 example:
    // A = [4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7]
    const std::vector<uint8_t> data = {4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7};
    WaveletMatrix<uint8_t> wm(data);

    // Subrange [4, 11) -> elements [3, 2, 1, 0, 2, 1, 4]
    // Sorted: [0, 1, 1, 2, 2, 3, 4]
    EXPECT_EQ(wm.range_quantile(4, 11, 0), 0);
    EXPECT_EQ(wm.range_quantile(4, 11, 1), 1);
    EXPECT_EQ(wm.range_quantile(4, 11, 2), 1);
    EXPECT_EQ(wm.range_quantile(4, 11, 3), 2);
    EXPECT_EQ(wm.range_quantile(4, 11, 4), 2);
    EXPECT_EQ(wm.range_quantile(4, 11, 5), 3);
    EXPECT_EQ(wm.range_quantile(4, 11, 6), 4);

    // Full range [0, 13)
    // Sorted: [0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7]
    EXPECT_EQ(wm.range_quantile(0, 13, 0), 0);  // min
    EXPECT_EQ(wm.range_quantile(0, 13, 12), 7); // max
    EXPECT_EQ(wm.range_quantile(0, 13, 6), 3);  // median
}

TEST(WaveletMatrixTest, RangeQuantileRandomSubranges) {
    std::mt19937_64 rng(2024);
    const size_t n = 250;
    std::vector<uint32_t> data(n);
    std::uniform_int_distribution<uint32_t> val_dist(0, 1000);
    for (size_t i = 0; i < n; ++i) {
        data[i] = val_dist(rng);
    }

    WaveletMatrix<uint32_t> wm(data);

    std::uniform_int_distribution<size_t> pos_dist(0, n);
    for (size_t trial = 0; trial < 100; ++trial) {
        size_t l = pos_dist(rng);
        size_t r = pos_dist(rng);
        if (l >= r) {
            std::swap(l, r);
            if (l == r) {
                if (r < n)
                    ++r;
                else if (l > 0)
                    --l;
                else
                    continue;
            }
        }

        std::vector<uint32_t> sub(data.begin() + l, data.begin() + r);
        std::sort(sub.begin(), sub.end());

        // Check min, max, median, and random k
        EXPECT_EQ(wm.range_quantile(l, r, 0), sub[0]);
        EXPECT_EQ(wm.range_quantile(l, r, sub.size() - 1), sub.back());
        size_t mid_k = sub.size() / 2;
        EXPECT_EQ(wm.range_quantile(l, r, mid_k), sub[mid_k]);

        std::uniform_int_distribution<size_t> k_dist(0, sub.size() - 1);
        size_t rand_k = k_dist(rng);
        EXPECT_EQ(wm.range_quantile(l, r, rand_k), sub[rand_k]);
    }
}

TEST(WaveletMatrixTest, SelectBasicAndPaperExample) {
    // Claude & Navarro (2015) Fig. 4 example:
    // A = [4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7]
    const std::vector<uint8_t> data = {4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7};
    WaveletMatrix<uint8_t> wm(data);

    // 1-indexed occurrences of character 1: indices 6, 9, 11
    EXPECT_EQ(wm.select(1, 1), 6ULL);
    EXPECT_EQ(wm.select(1, 2), 9ULL);
    EXPECT_EQ(wm.select(1, 3), 11ULL);
    EXPECT_EQ(wm.select(1, 0), data.size()); // k=0 out of bounds
    EXPECT_EQ(wm.select(1, 4), data.size()); // k > total occurrences

    // Occurrences of character 4: indices 0, 10
    EXPECT_EQ(wm.select(4, 1), 0ULL);
    EXPECT_EQ(wm.select(4, 2), 10ULL);
    EXPECT_EQ(wm.select(4, 3), data.size());

    // Character 0: index 7
    EXPECT_EQ(wm.select(0, 1), 7ULL);
    EXPECT_EQ(wm.select(0, 2), data.size());

    // Non-existent characters
    EXPECT_EQ(wm.select(8, 1), data.size());
    EXPECT_EQ(wm.select(255, 1), data.size());
}

TEST(WaveletMatrixTest, SelectExhaustiveRandom) {
    std::mt19937_64 rng(999);
    const size_t n = 300;
    std::vector<uint16_t> data(n);
    std::uniform_int_distribution<uint16_t> dist(0, 40);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    WaveletMatrix<uint16_t> wm(data);

    // For every character, collect positions by linear scan and test select
    for (uint16_t c = 0; c <= 45; ++c) {
        std::vector<size_t> expected_positions;
        for (size_t i = 0; i < n; ++i) {
            if (data[i] == c) {
                expected_positions.push_back(i);
            }
        }

        for (size_t k = 1; k <= expected_positions.size(); ++k) {
            size_t pos = wm.select(c, k);
            EXPECT_EQ(pos, expected_positions[k - 1]) << "Mismatch for char " << c << ", k=" << k;
            EXPECT_EQ(wm.access(pos), c);
            EXPECT_EQ(wm.rank(c, pos + 1), k);
        }

        // Test boundary k = expected_positions.size() + 1
        EXPECT_EQ(wm.select(c, expected_positions.size() + 1), n);
    }
}

TEST(WaveletMatrixTest, RankBasicAndPaperExample) {
    // Claude & Navarro (2015) Fig. 4 example:
    // A = [4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7]
    const std::vector<uint8_t> data = {4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7};
    WaveletMatrix<uint8_t> wm(data);

    // Character 1 appears at indices 6, 9, 11
    EXPECT_EQ(wm.rank(1, 0), 0ULL);
    EXPECT_EQ(wm.rank(1, 6), 0ULL);
    EXPECT_EQ(wm.rank(1, 7), 1ULL);
    EXPECT_EQ(wm.rank(1, 9), 1ULL);
    EXPECT_EQ(wm.rank(1, 10), 2ULL);
    EXPECT_EQ(wm.rank(1, 11), 2ULL);
    EXPECT_EQ(wm.rank(1, 12), 3ULL);
    EXPECT_EQ(wm.rank(1, 13), 3ULL);

    // Character 4 appears at indices 0, 10
    EXPECT_EQ(wm.rank(4, 0), 0ULL);
    EXPECT_EQ(wm.rank(4, 1), 1ULL);
    EXPECT_EQ(wm.rank(4, 10), 1ULL);
    EXPECT_EQ(wm.rank(4, 11), 2ULL);
    EXPECT_EQ(wm.rank(4, 13), 2ULL);

    // Characters not in sequence
    EXPECT_EQ(wm.rank(8, 5), 0ULL);
    EXPECT_EQ(wm.rank(255, 13), 0ULL);
}

TEST(WaveletMatrixTest, RankExhaustiveRandom) {
    std::mt19937_64 rng(42);
    const size_t n = 200;
    std::vector<uint16_t> data(n);
    std::uniform_int_distribution<uint16_t> dist(0, 30);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    WaveletMatrix<uint16_t> wm(data);

    for (uint16_t c = 0; c <= 35; ++c) {
        for (size_t i = 0; i <= n; ++i) {
            size_t expected = std::count(data.begin(), data.begin() + i, c);
            EXPECT_EQ(wm.rank(c, i), expected) << "Mismatch for char " << c << " at prefix " << i;
        }
    }
}

TEST(WaveletMatrixTest, EmptySequence) {
    std::vector<uint32_t> empty_data;
    WaveletMatrix<uint32_t> wm(empty_data);

    EXPECT_EQ(wm.size(), 0ULL);
    EXPECT_EQ(wm.sigma(), 0ULL);
    EXPECT_GT(wm.memory_usage_bytes(), 0ULL);
}

TEST(WaveletMatrixTest, ClaudeNavarroPaperExampleAccess) {
    // Claude & Navarro (2015) Fig. 4 example:
    // A = [4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7]
    const std::vector<uint8_t> data = {4, 7, 6, 5, 3, 2, 1, 0, 2, 1, 4, 1, 7};
    WaveletMatrix<uint8_t> wm(data);

    EXPECT_EQ(wm.size(), data.size());
    EXPECT_EQ(wm.sigma(), 8ULL); // max element is 7, so sigma = 8

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(wm.access(i), data[i]) << "Mismatch at index " << i;
    }
}

TEST(WaveletMatrixTest, SingleElement) {
    const std::vector<uint16_t> data = {42};
    WaveletMatrix<uint16_t> wm(data);

    EXPECT_EQ(wm.size(), 1ULL);
    EXPECT_EQ(wm.sigma(), 43ULL);
    EXPECT_EQ(wm.access(0), 42);
}

TEST(WaveletMatrixTest, AllIdenticalElements) {
    const std::vector<uint32_t> data(100, 15);
    WaveletMatrix<uint32_t> wm(data);

    EXPECT_EQ(wm.size(), 100ULL);
    EXPECT_EQ(wm.sigma(), 16ULL);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(wm.access(i), 15);
    }
}

TEST(WaveletMatrixTest, RandomAccessLarge) {
    std::mt19937_64 rng(1337);
    const size_t n = 2000;
    std::vector<uint32_t> data(n);
    std::uniform_int_distribution<uint32_t> dist(0, 500);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }

    WaveletMatrix<uint32_t> wm(data);
    EXPECT_EQ(wm.size(), n);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(wm.access(i), data[i]) << "Failed at index " << i;
    }
}
