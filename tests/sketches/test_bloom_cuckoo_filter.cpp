#include "algoat/sketches/bloom_cuckoo_filter.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// Slice 1: Concepts, DefaultHasher, BlockedBloomFilter Sizing & Configuration
// ============================================================================

TEST(BlockedBloomFilterTest, FilterHasherConcept) {
    static_assert(FilterHasher<DefaultHasher, uint64_t>);
    static_assert(FilterHasher<DefaultHasher, std::string>);
    static_assert(FilterHasher<DefaultHasher, std::string_view>);

    struct CustomGoodHasher {
        uint64_t operator()(uint32_t x) const noexcept {
            return static_cast<uint64_t>(x) * 0x9e3779b97f4a7c15ULL;
        }
    };
    static_assert(FilterHasher<CustomGoodHasher, uint32_t>);

    struct BadHasher {
        void operator()(int) const {}
    };
    static_assert(!FilterHasher<BadHasher, int>);
}

TEST(BlockedBloomFilterTest, CacheLineBlockAlignment) {
    using Block = BlockedBloomFilter<uint64_t>::CacheLineBlock;
    static_assert(sizeof(Block) == 64, "CacheLineBlock must be exactly 64 bytes (512 bits)");
    static_assert(alignof(Block) == 64, "CacheLineBlock must be aligned to 64 bytes");
}

TEST(BlockedBloomFilterTest, ConstructionAndSizingTargetFPR) {
    // 1000 items, target_fpr = 0.01 (~10 bits/item -> ~10,000 bits -> ~20 blocks -> bit_ceil = 32
    // blocks)
    BlockedBloomFilter<uint64_t> filter(1000, 0.01);
    EXPECT_GT(filter.num_blocks(), 0U);
    EXPECT_EQ(filter.memory_usage_bytes(), filter.num_blocks() * 64U);
    EXPECT_EQ(filter.theoretical_fpr(), 0.0); // 0 items inserted
}

TEST(BlockedBloomFilterTest, ConstructionAndSizingExplicitBlocks) {
    BlockedBloomFilter<uint64_t> filter(16, 10);
    EXPECT_EQ(filter.num_blocks(), 16U);
    EXPECT_EQ(filter.memory_usage_bytes(), 16U * 64U);
}

TEST(BlockedBloomFilterTest, ClearResetsState) {
    BlockedBloomFilter<uint64_t> filter(8, 8);
    EXPECT_EQ(filter.num_blocks(), 8U);
    filter.clear();
    EXPECT_EQ(filter.num_blocks(), 8U);
    EXPECT_EQ(filter.theoretical_fpr(), 0.0);
}

// ============================================================================
// Slice 2: BlockedBloomFilter Insert, Contains, and Prefetched Batch Queries
// ============================================================================

TEST(BlockedBloomFilterTest, BasicInsertAndContains) {
    BlockedBloomFilter<std::string> filter(100, 0.01);
    EXPECT_FALSE(filter.contains("alpha"));
    filter.insert("alpha");
    EXPECT_TRUE(filter.contains("alpha"));
    EXPECT_FALSE(filter.contains("beta"));
}

TEST(BlockedBloomFilterTest, ZeroFalseNegatives) {
    constexpr size_t N = 10000;
    BlockedBloomFilter<uint64_t> filter(N, 0.01);

    for (size_t i = 0; i < N; ++i) {
        filter.insert(i * 2 + 1); // odd numbers
    }

    // Every inserted item MUST return true (zero false negatives)
    for (size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(filter.contains(i * 2 + 1));
    }
}

TEST(BlockedBloomFilterTest, EmpiricalFPRWithinBound) {
    constexpr size_t N = 20000;
    BlockedBloomFilter<uint64_t> filter(N, 0.01);

    for (size_t i = 0; i < N; ++i) {
        filter.insert(i * 2); // even numbers
    }

    size_t false_positives = 0;
    constexpr size_t query_count = 50000;
    for (size_t i = 0; i < query_count; ++i) {
        uint64_t test_key = i * 2 + 1; // odd numbers (never inserted)
        if (filter.contains(test_key)) {
            ++false_positives;
        }
    }

    double empirical_fpr = static_cast<double>(false_positives) / static_cast<double>(query_count);
    // Target FPR is 0.01; empirical FPR should be close (allowing moderate tolerance for blocked
    // filter variance)
    EXPECT_LE(empirical_fpr, 0.015);
    EXPECT_GT(filter.theoretical_fpr(), 0.0);
}

TEST(BlockedBloomFilterTest, BatchContainsMatchesScalar) {
    constexpr size_t N = 1000;
    BlockedBloomFilter<uint64_t> filter(N, 0.01);

    for (size_t i = 0; i < N; ++i) {
        filter.insert(i * 3);
    }

    std::vector<uint64_t> query_keys(500);
    for (size_t i = 0; i < 500; ++i) {
        query_keys[i] = i * 2; // mixed present / absent
    }

    std::array<bool, 500> batch_results{};
    filter.contains_batch(query_keys, batch_results);

    for (size_t i = 0; i < 500; ++i) {
        bool expected = filter.contains(query_keys[i]);
        EXPECT_EQ(batch_results[i], expected);
    }
}

// ============================================================================
// Slice 3: CuckooFilter Template Constraints, Construction & Sizing Invariants
// ============================================================================

TEST(CuckooFilterTest, ConstructionAndSizing) {
    // 8-bit fingerprint, 4-slot bucket
    CuckooFilter<uint64_t, 8, 4> cf8_4(1000);
    EXPECT_EQ(cf8_4.size(), 0U);
    EXPECT_GE(cf8_4.capacity(), 1000U);
    EXPECT_EQ(cf8_4.load_factor(), 0.0);
    EXPECT_GT(cf8_4.memory_usage_bytes(), 0U);
    EXPECT_EQ((CuckooFilter<uint64_t, 8, 4>::kMaxKicks), 500U);

    // 16-bit fingerprint, 8-slot bucket
    CuckooFilter<std::string, 16, 8> cf16_8(5000);
    EXPECT_EQ(cf16_8.size(), 0U);
    EXPECT_GE(cf16_8.capacity(), 5000U);
    EXPECT_EQ(cf16_8.load_factor(), 0.0);
    EXPECT_GT(cf16_8.memory_usage_bytes(), 0U);
}

TEST(CuckooFilterTest, PowerOfTwoBucketsConstraint) {
    CuckooFilter<uint64_t, 8, 4> filter(100);
    // 100 items / 4 slots = 25 buckets -> bit_ceil is 32 buckets
    EXPECT_EQ(filter.capacity(), 32U * 4U);
    EXPECT_EQ(filter.memory_usage_bytes(), 32U * 4U * sizeof(uint8_t));
}

// ============================================================================
// Slice 4: CuckooFilter Insert, Vectorized Contains & High Load Factor
// ============================================================================

TEST(CuckooFilterTest, BasicInsertAndContains) {
    CuckooFilter<std::string, 8, 4> filter(64);
    EXPECT_FALSE(filter.contains("item1"));
    EXPECT_TRUE(filter.insert("item1"));
    EXPECT_TRUE(filter.contains("item1"));
    EXPECT_FALSE(filter.contains("item2"));
    EXPECT_EQ(filter.size(), 1U);
}

TEST(CuckooFilterTest, ZeroFalseNegatives) {
    constexpr size_t N = 1000;
    CuckooFilter<uint64_t, 8, 4> filter(2048);

    for (size_t i = 0; i < N; ++i) {
        ASSERT_TRUE(filter.insert(i * 10 + 1));
    }

    for (size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(filter.contains(i * 10 + 1));
    }
    EXPECT_EQ(filter.size(), N);
}

TEST(CuckooFilterTest, EmpiricalFPRWithinBounds8BitAnd16Bit) {
    // 8-bit fingerprint test
    {
        constexpr size_t N = 10000;
        CuckooFilter<uint64_t, 8, 4> cf8(20000);
        for (size_t i = 0; i < N; ++i) {
            ASSERT_TRUE(cf8.insert(i * 2));
        }

        size_t fp_count = 0;
        constexpr size_t queries = 50000;
        for (size_t i = 0; i < queries; ++i) {
            uint64_t key = i * 2 + 1;
            if (cf8.contains(key)) {
                ++fp_count;
            }
        }
        double fpr8 = static_cast<double>(fp_count) / static_cast<double>(queries);
        EXPECT_LE(fpr8, 0.035);
    }

    // 12-bit fingerprint test
    {
        constexpr size_t N = 10000;
        CuckooFilter<uint64_t, 12, 4> cf12(20000);
        for (size_t i = 0; i < N; ++i) {
            ASSERT_TRUE(cf12.insert(i * 2));
        }

        size_t fp_count = 0;
        constexpr size_t queries = 50000;
        for (size_t i = 0; i < queries; ++i) {
            uint64_t key = i * 2 + 1;
            if (cf12.contains(key)) {
                ++fp_count;
            }
        }
        double fpr12 = static_cast<double>(fp_count) / static_cast<double>(queries);
        EXPECT_LE(fpr12, 0.003);
    }

    // 16-bit fingerprint test
    {
        constexpr size_t N = 10000;
        CuckooFilter<uint64_t, 16, 4> cf16(20000);
        for (size_t i = 0; i < N; ++i) {
            ASSERT_TRUE(cf16.insert(i * 2));
        }

        size_t fp_count = 0;
        constexpr size_t queries = 50000;
        for (size_t i = 0; i < queries; ++i) {
            uint64_t key = i * 2 + 1;
            if (cf16.contains(key)) {
                ++fp_count;
            }
        }
        double fpr16 = static_cast<double>(fp_count) / static_cast<double>(queries);
        EXPECT_LE(fpr16, 0.002);
    }
}

TEST(CuckooFilterTest, HighLoadFactorKicking) {
    // b=4 canonical cuckoo filter achieves >= 95% load factor
    constexpr size_t target_capacity = 4096; // 1024 buckets * 4 slots
    CuckooFilter<uint64_t, 16, 4> filter(target_capacity);

    size_t inserted = 0;
    for (size_t i = 0; i < target_capacity; ++i) {
        if (!filter.insert(i + 1)) {
            break;
        }
        ++inserted;
    }

    double achieved_load = filter.load_factor();
    EXPECT_GE(achieved_load, 0.95);
    EXPECT_EQ(filter.size(), inserted);

    // Verify all inserted items are contained
    for (size_t i = 0; i < inserted; ++i) {
        EXPECT_TRUE(filter.contains(i + 1));
    }
}

// ============================================================================
// Slice 5: CuckooFilter Dynamic Deletion (remove)
// ============================================================================

TEST(CuckooFilterTest, RemoveBasic) {
    CuckooFilter<std::string, 8, 4> filter(64);
    EXPECT_TRUE(filter.insert("cat"));
    EXPECT_TRUE(filter.insert("dog"));

    EXPECT_TRUE(filter.contains("cat"));
    EXPECT_TRUE(filter.contains("dog"));
    EXPECT_EQ(filter.size(), 2U);

    EXPECT_TRUE(filter.remove("cat"));
    EXPECT_FALSE(filter.contains("cat"));
    EXPECT_TRUE(filter.contains("dog"));
    EXPECT_EQ(filter.size(), 1U);

    // Deleting again returns false
    EXPECT_FALSE(filter.remove("cat"));
    EXPECT_FALSE(filter.remove("bird")); // never inserted
}

TEST(CuckooFilterTest, RemoveDuplicateItems) {
    CuckooFilter<uint64_t, 16, 4> filter(64);
    uint64_t key = 42;

    EXPECT_TRUE(filter.insert(key));
    EXPECT_TRUE(filter.insert(key));
    EXPECT_EQ(filter.size(), 2U);
    EXPECT_TRUE(filter.contains(key));

    EXPECT_TRUE(filter.remove(key));
    EXPECT_EQ(filter.size(), 1U);
    EXPECT_TRUE(filter.contains(key)); // second occurrence remains

    EXPECT_TRUE(filter.remove(key));
    EXPECT_EQ(filter.size(), 0U);
    EXPECT_FALSE(filter.contains(key)); // both removed
}

TEST(CuckooFilterTest, DynamicInterleavedInsertRemove) {
    CuckooFilter<uint64_t, 16, 4> filter(256);

    for (size_t iter = 0; iter < 50; ++iter) {
        for (size_t i = 0; i < 20; ++i) {
            uint64_t key = iter * 1000 + i;
            EXPECT_TRUE(filter.insert(key));
        }
        for (size_t i = 0; i < 20; ++i) {
            uint64_t key = iter * 1000 + i;
            EXPECT_TRUE(filter.remove(key));
        }
        for (size_t i = 0; i < 20; ++i) {
            uint64_t key = iter * 1000 + i;
            EXPECT_FALSE(filter.contains(key));
        }
        EXPECT_EQ(filter.size(), 0U);
    }
}

// ============================================================================
// Zero-Allocation Guarantee Invariants
// ============================================================================

TEST(ZeroAllocationTest, MethodsAreGuaranteedNoexcept) {
    BlockedBloomFilter<uint64_t> bbf(100, 0.01);
    CuckooFilter<uint64_t, 8, 4> cf(100);

    uint64_t key = 42;
    static_assert(noexcept(bbf.insert(key)));
    static_assert(noexcept(bbf.contains(key)));
    static_assert(noexcept(cf.contains(key)));
    static_assert(noexcept(cf.remove(key)));

    std::span<const uint64_t> keys_span(&key, 1);
    bool res = false;
    std::span<bool> res_span(&res, 1);
    static_assert(noexcept(bbf.contains_batch(keys_span, res_span)));
}
