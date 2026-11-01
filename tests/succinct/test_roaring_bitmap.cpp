#include <algoat/succinct/roaring_bitmap.hpp>
#include <gtest/gtest.h>
#include <ranges>
#include <unordered_set>
#include <vector>

using namespace algoat::succinct;

// Static concept assertion
static_assert(std::ranges::forward_range<RoaringBitmap>);

TEST(RoaringBitmapTest, DefaultConstructionIsEmpty) {
    RoaringBitmap bm;
    EXPECT_TRUE(bm.empty());
    EXPECT_EQ(bm.cardinality(), 0ULL);
    EXPECT_FALSE(bm.contains(0));
    EXPECT_FALSE(bm.contains(42));
    EXPECT_FALSE(bm.contains(65536));
    EXPECT_FALSE(bm.contains(1000000));
}

TEST(RoaringBitmapTest, AddAndContainsSingleChunk) {
    RoaringBitmap bm;
    EXPECT_TRUE(bm.add(10));
    EXPECT_TRUE(bm.add(20));
    EXPECT_TRUE(bm.add(30));

    // Duplicate add returns false
    EXPECT_FALSE(bm.add(10));
    EXPECT_FALSE(bm.add(20));

    EXPECT_FALSE(bm.empty());
    EXPECT_EQ(bm.cardinality(), 3ULL);
    EXPECT_TRUE(bm.contains(10));
    EXPECT_TRUE(bm.contains(20));
    EXPECT_TRUE(bm.contains(30));
    EXPECT_FALSE(bm.contains(15));
    EXPECT_FALSE(bm.contains(0));
}

TEST(RoaringBitmapTest, InitializerListAndSpanConstruction) {
    RoaringBitmap bm1{1, 5, 10, 50, 100, 65536, 131072};
    EXPECT_EQ(bm1.cardinality(), 7ULL);
    EXPECT_TRUE(bm1.contains(1));
    EXPECT_TRUE(bm1.contains(65536));
    EXPECT_TRUE(bm1.contains(131072));
    EXPECT_FALSE(bm1.contains(2));

    std::vector<uint32_t> sorted_vals = {10, 20, 30, 40, 50};
    RoaringBitmap bm2(std::span<const uint32_t>{sorted_vals});
    EXPECT_EQ(bm2.cardinality(), 5ULL);
    EXPECT_TRUE(bm2.contains(10));
    EXPECT_TRUE(bm2.contains(50));
    EXPECT_FALSE(bm2.contains(15));
}

TEST(RoaringBitmapTest, RemoveElements) {
    RoaringBitmap bm{10, 20, 30};
    EXPECT_TRUE(bm.remove(20));
    EXPECT_FALSE(bm.remove(20)); // Already removed
    EXPECT_FALSE(bm.remove(99)); // Never existed

    EXPECT_EQ(bm.cardinality(), 2ULL);
    EXPECT_TRUE(bm.contains(10));
    EXPECT_FALSE(bm.contains(20));
    EXPECT_TRUE(bm.contains(30));

    EXPECT_TRUE(bm.remove(10));
    EXPECT_TRUE(bm.remove(30));
    EXPECT_TRUE(bm.empty());
    EXPECT_EQ(bm.cardinality(), 0ULL);
}

TEST(RoaringBitmapTest, ContainerTransitionsArrayToBitsetAndBack) {
    RoaringBitmap bm;
    // Add 4096 elements to chunk 0 (values 0 .. 4095)
    for (uint32_t i = 0; i < 4096; ++i) {
        EXPECT_TRUE(bm.add(i * 2)); // 0, 2, 4, ... 8190
    }
    EXPECT_EQ(bm.cardinality(), 4096ULL);
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(8190));
    EXPECT_FALSE(bm.contains(1));

    // 4097-th element triggers Array -> Bitset transition
    EXPECT_TRUE(bm.add(1));
    EXPECT_EQ(bm.cardinality(), 4097ULL);
    EXPECT_TRUE(bm.contains(1));
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(8190));
    EXPECT_FALSE(bm.contains(3));

    // Remove elements down to 4096 (triggers Bitset -> Array transition)
    EXPECT_TRUE(bm.remove(1));
    EXPECT_EQ(bm.cardinality(), 4096ULL);
    EXPECT_FALSE(bm.contains(1));
    EXPECT_TRUE(bm.contains(0));
    EXPECT_TRUE(bm.contains(8190));

    // Further removal works in ArrayContainer
    EXPECT_TRUE(bm.remove(0));
    EXPECT_EQ(bm.cardinality(), 4095ULL);
    EXPECT_FALSE(bm.contains(0));
}

TEST(RoaringBitmapTest, MultiChunkHandling) {
    RoaringBitmap bm;
    bm.add(10);             // chunk 0
    bm.add(65536 + 10);     // chunk 1
    bm.add(2 * 65536 + 10); // chunk 2
    bm.add(100 * 65536 + 5);

    EXPECT_EQ(bm.cardinality(), 4ULL);
    EXPECT_TRUE(bm.contains(10));
    EXPECT_TRUE(bm.contains(65536 + 10));
    EXPECT_TRUE(bm.contains(2 * 65536 + 10));
    EXPECT_TRUE(bm.contains(100 * 65536 + 5));
    EXPECT_FALSE(bm.contains(65536));
    EXPECT_FALSE(bm.contains(100 * 65536 + 6));
}

TEST(RoaringBitmapTest, RangeIterationAndPipelines) {
    std::vector<uint32_t> expected = {10, 20, 30, 65540, 65545, 131072, 131073, 131074};
    RoaringBitmap bm;
    for (uint32_t v : expected) {
        bm.add(v);
    }

    std::vector<uint32_t> actual;
    for (uint32_t val : bm) {
        actual.push_back(val);
    }
    EXPECT_EQ(actual, expected);

    // Range pipeline views
    auto even_view = bm | std::views::filter([](uint32_t x) { return x % 2 == 0; });
    std::vector<uint32_t> evens;
    for (uint32_t v : even_view) {
        evens.push_back(v);
    }
    std::vector<uint32_t> expected_evens = {10, 20, 30, 65540, 131072, 131074};
    EXPECT_EQ(evens, expected_evens);
}

TEST(RoaringBitmapTest, BitsetIteration) {
    RoaringBitmap bm;
    std::vector<uint32_t> vals;
    for (uint32_t i = 0; i < 5000; ++i) {
        vals.push_back(i * 3);
        bm.add(i * 3);
    }

    std::vector<uint32_t> collected;
    for (uint32_t v : bm) {
        collected.push_back(v);
    }
    EXPECT_EQ(collected, vals);
}

TEST(RoaringBitmapTest, RunOptimizationAndIteration) {
    RoaringBitmap bm;
    // Add consecutive range [1000, 5000] and [70000, 75000]
    for (uint32_t i = 1000; i <= 5000; ++i) {
        bm.add(i);
    }
    for (uint32_t i = 70000; i <= 75000; ++i) {
        bm.add(i);
    }

    const uint64_t card_before = bm.cardinality();
    EXPECT_EQ(card_before, (5000 - 1000 + 1) + (75000 - 70000 + 1));

    bm.run_optimize();

    EXPECT_EQ(bm.cardinality(), card_before);
    EXPECT_TRUE(bm.contains(1000));
    EXPECT_TRUE(bm.contains(3000));
    EXPECT_TRUE(bm.contains(5000));
    EXPECT_TRUE(bm.contains(70000));
    EXPECT_TRUE(bm.contains(75000));
    EXPECT_FALSE(bm.contains(999));
    EXPECT_FALSE(bm.contains(5001));

    // Verify iteration over run containers
    uint32_t count = 0;
    for (uint32_t v : bm) {
        EXPECT_TRUE(bm.contains(v));
        ++count;
    }
    EXPECT_EQ(count, card_before);
}

TEST(RoaringBitmapTest, BitwiseSetOperations) {
    RoaringBitmap a{1, 2, 3, 4, 10, 20, 65536 + 1, 65536 + 2};
    RoaringBitmap b{3, 4, 5, 6, 20, 30, 65536 + 2, 65536 + 3};

    // Intersection
    RoaringBitmap inter = a & b;
    EXPECT_EQ(inter.cardinality(), 4ULL); // 3, 4, 20, 65536+2
    EXPECT_TRUE(inter.contains(3));
    EXPECT_TRUE(inter.contains(4));
    EXPECT_TRUE(inter.contains(20));
    EXPECT_TRUE(inter.contains(65536 + 2));
    EXPECT_FALSE(inter.contains(1));

    // Union
    RoaringBitmap un = a | b;
    EXPECT_EQ(un.cardinality(), 12ULL);
    EXPECT_TRUE(un.contains(1));
    EXPECT_TRUE(un.contains(6));
    EXPECT_TRUE(un.contains(30));

    // Difference
    RoaringBitmap diff = a - b;
    EXPECT_EQ(diff.cardinality(), 4ULL); // 1, 2, 10, 65536+1
    EXPECT_TRUE(diff.contains(1));
    EXPECT_TRUE(diff.contains(2));
    EXPECT_TRUE(diff.contains(10));
    EXPECT_TRUE(diff.contains(65536 + 1));
    EXPECT_FALSE(diff.contains(3));

    // Symmetric Difference (XOR)
    RoaringBitmap sym_diff = a ^ b;
    EXPECT_EQ(sym_diff.cardinality(),
              8ULL); // 1, 2, 5, 6, 10, 30, 65536+1, 65536+3... wait: (10 - 4 = 6)
    EXPECT_TRUE(sym_diff.contains(1));
    EXPECT_TRUE(sym_diff.contains(5));
    EXPECT_FALSE(sym_diff.contains(3));
    EXPECT_FALSE(sym_diff.contains(20));

    // In-place operations
    RoaringBitmap a_copy = a;
    a_copy &= b;
    EXPECT_EQ(a_copy.cardinality(), inter.cardinality());

    a_copy = a;
    a_copy |= b;
    EXPECT_EQ(a_copy.cardinality(), un.cardinality());

    a_copy = a;
    a_copy -= b;
    EXPECT_EQ(a_copy.cardinality(), diff.cardinality());

    a_copy = a;
    a_copy ^= b;
    EXPECT_EQ(a_copy.cardinality(), sym_diff.cardinality());
}

TEST(RoaringBitmapTest, BitsetContainerBitwiseOperations) {
    RoaringBitmap a;
    RoaringBitmap b;
    for (uint32_t i = 0; i < 5000; ++i) {
        a.add(i);        // [0, 4999] -> BitsetContainer
        b.add(i + 2500); // [2500, 7499] -> BitsetContainer
    }

    RoaringBitmap inter = a & b; // [2500, 4999] -> 2500 elements (downcasts to ArrayContainer!)
    EXPECT_EQ(inter.cardinality(), 2500ULL);
    EXPECT_TRUE(inter.contains(2500));
    EXPECT_TRUE(inter.contains(4999));
    EXPECT_FALSE(inter.contains(2499));
    EXPECT_FALSE(inter.contains(5000));

    RoaringBitmap un = a | b; // [0, 7499] -> 7500 elements (BitsetContainer)
    EXPECT_EQ(un.cardinality(), 7500ULL);
    EXPECT_TRUE(un.contains(0));
    EXPECT_TRUE(un.contains(7499));

    RoaringBitmap diff = a - b; // [0, 2499] -> 2500 elements (ArrayContainer)
    EXPECT_EQ(diff.cardinality(), 2500ULL);
    EXPECT_TRUE(diff.contains(0));
    EXPECT_TRUE(diff.contains(2499));
    EXPECT_FALSE(diff.contains(2500));
}

TEST(RoaringBitmapTest, JaccardIndexCalculation) {
    RoaringBitmap empty1;
    RoaringBitmap empty2;
    EXPECT_DOUBLE_EQ(RoaringBitmap::jaccard_index(empty1, empty2), 1.0);

    RoaringBitmap a{1, 2, 3, 4};
    RoaringBitmap b{3, 4, 5, 6};
    // Intersection: {3, 4} (size 2)
    // Union: {1, 2, 3, 4, 5, 6} (size 6)
    // Jaccard: 2 / 6 = 1.0 / 3.0
    EXPECT_NEAR(RoaringBitmap::jaccard_index(a, b), 1.0 / 3.0, 1e-9);

    RoaringBitmap disjoint{10, 11, 12};
    EXPECT_DOUBLE_EQ(RoaringBitmap::jaccard_index(a, disjoint), 0.0);
    EXPECT_DOUBLE_EQ(RoaringBitmap::jaccard_index(a, a), 1.0);
}

TEST(RoaringBitmapTest, SerializationAndDeserializationRoundTrip) {
    // Empty bitmap
    RoaringBitmap empty;
    auto empty_bytes = empty.serialize();
    RoaringBitmap empty_deser = RoaringBitmap::deserialize(empty_bytes);
    EXPECT_TRUE(empty_deser.empty());

    // Bitmap with Array and Bitset containers (no run)
    RoaringBitmap bm;
    for (uint32_t i = 0; i < 100; ++i) {
        bm.add(i * 10);
    }
    for (uint32_t i = 0; i < 5000; ++i) {
        bm.add(65536 + i);
    }
    bm.add(2 * 65536 + 42);

    auto bytes1 = bm.serialize();
    RoaringBitmap deser1 = RoaringBitmap::deserialize(bytes1);
    EXPECT_EQ(deser1.cardinality(), bm.cardinality());
    for (uint32_t v : bm) {
        EXPECT_TRUE(deser1.contains(v));
    }

    // Bitmap with Run containers (after run_optimize)
    bm.run_optimize();
    auto bytes2 = bm.serialize();
    RoaringBitmap deser2 = RoaringBitmap::deserialize(bytes2);
    EXPECT_EQ(deser2.cardinality(), bm.cardinality());
    for (uint32_t v : bm) {
        EXPECT_TRUE(deser2.contains(v));
    }

    // Error handling on corrupt buffer
    std::vector<uint8_t> corrupt_bytes = {0x00, 0x01};
    EXPECT_THROW(RoaringBitmap::deserialize(corrupt_bytes), std::runtime_error);
}

TEST(RoaringBitmapTest, MemoryUsageReporting) {
    RoaringBitmap bm;
    EXPECT_GT(bm.memory_usage_bytes(), 0ULL);
    bm.add(10);
    EXPECT_GT(bm.memory_usage_bytes(), sizeof(RoaringBitmap));
}

TEST(RoaringBitmapTest, HighVolumeReferenceSetEquivalence10M) {
    RoaringBitmap bm;
    std::unordered_set<uint32_t> ref;

    // Fast deterministic LCG
    uint64_t state = 123456789ULL;
    auto lcg_rand = [&]() noexcept -> uint32_t {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 32);
    };

    // Phase 1: 2.5 million pseudo-random insertions
    for (size_t i = 0; i < 2500000; ++i) {
        uint32_t val = lcg_rand() % 5000000;
        bool b_res = bm.add(val);
        bool r_res = ref.insert(val).second;
        EXPECT_EQ(b_res, r_res);
    }
    EXPECT_EQ(bm.cardinality(), ref.size());

    // Phase 2: 2.5 million pseudo-random membership queries
    for (size_t i = 0; i < 2500000; ++i) {
        uint32_t val = lcg_rand() % 5000000;
        EXPECT_EQ(bm.contains(val), (ref.find(val) != ref.end()));
    }

    // Phase 3: 2.5 million pseudo-random deletions
    for (size_t i = 0; i < 2500000; ++i) {
        uint32_t val = lcg_rand() % 5000000;
        bool b_res = bm.remove(val);
        bool r_res = (ref.erase(val) > 0);
        EXPECT_EQ(b_res, r_res);
    }
    EXPECT_EQ(bm.cardinality(), ref.size());

    // Phase 4: 2.5 million secondary insertions & set operations (10M cumulative)
    RoaringBitmap bm2;
    std::unordered_set<uint32_t> ref2;
    for (size_t i = 0; i < 2500000; ++i) {
        uint32_t val = lcg_rand() % 5000000;
        bm2.add(val);
        ref2.insert(val);
    }
    EXPECT_EQ(bm2.cardinality(), ref2.size());

    RoaringBitmap inter = bm & bm2;
    size_t ref_inter_count = 0;
    for (uint32_t v : ref) {
        if (ref2.find(v) != ref2.end()) {
            ++ref_inter_count;
            EXPECT_TRUE(inter.contains(v));
        }
    }
    EXPECT_EQ(inter.cardinality(), ref_inter_count);

    RoaringBitmap un = bm | bm2;
    std::unordered_set<uint32_t> ref_un = ref;
    ref_un.insert(ref2.begin(), ref2.end());
    EXPECT_EQ(un.cardinality(), ref_un.size());
}
