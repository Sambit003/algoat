#include "algoat/searching/veb_tree.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <set>
#include <vector>

using namespace algoat::searching;

TEST(VEBTreeTest, BaseCase1Bit) {
    VanEmdeBoasTree<1> veb;
    EXPECT_TRUE(veb.empty());
    EXPECT_FALSE(veb.contains(0));
    EXPECT_FALSE(veb.contains(1));
    EXPECT_EQ(veb.min(), std::nullopt);
    EXPECT_EQ(veb.max(), std::nullopt);
    EXPECT_EQ(veb.predecessor(0), std::nullopt);
    EXPECT_EQ(veb.predecessor(1), std::nullopt);
    EXPECT_EQ(veb.successor(0), std::nullopt);
    EXPECT_EQ(veb.successor(1), std::nullopt);

    veb.insert(0);
    EXPECT_FALSE(veb.empty());
    EXPECT_TRUE(veb.contains(0));
    EXPECT_FALSE(veb.contains(1));
    EXPECT_EQ(veb.min(), 0);
    EXPECT_EQ(veb.max(), 0);

    veb.insert(1);
    EXPECT_TRUE(veb.contains(0));
    EXPECT_TRUE(veb.contains(1));
    EXPECT_EQ(veb.min(), 0);
    EXPECT_EQ(veb.max(), 1);

    EXPECT_EQ(veb.successor(0), 1);
    EXPECT_EQ(veb.successor(1), std::nullopt);
    EXPECT_EQ(veb.successor(2), std::nullopt);
    EXPECT_EQ(veb.predecessor(2), 1);
    EXPECT_EQ(veb.predecessor(1), 0);
    EXPECT_EQ(veb.predecessor(0), std::nullopt);
    EXPECT_TRUE(veb.find(0));
    EXPECT_TRUE(veb.find(1));

    veb.erase(0);
    EXPECT_FALSE(veb.contains(0));
    EXPECT_TRUE(veb.contains(1));
    EXPECT_EQ(veb.min(), 1);
    EXPECT_EQ(veb.max(), 1);
    EXPECT_EQ(veb.predecessor(2), 1);
    EXPECT_EQ(veb.predecessor(1), std::nullopt);

    veb.erase(1);
    EXPECT_TRUE(veb.empty());
    EXPECT_FALSE(veb.contains(1));
    EXPECT_EQ(veb.min(), std::nullopt);
    EXPECT_EQ(veb.max(), std::nullopt);
}

TEST(VEBTreeTest, SmallUniverse4Bit) {
    VanEmdeBoasTree<4> veb; // universe [0, 15]
    EXPECT_TRUE(veb.empty());

    veb.insert(5);
    veb.insert(2);
    veb.insert(13);
    veb.insert(8);

    EXPECT_FALSE(veb.empty());
    EXPECT_EQ(veb.min(), 2);
    EXPECT_EQ(veb.max(), 13);

    EXPECT_TRUE(veb.contains(2));
    EXPECT_TRUE(veb.contains(5));
    EXPECT_TRUE(veb.contains(8));
    EXPECT_TRUE(veb.contains(13));
    EXPECT_FALSE(veb.contains(0));
    EXPECT_FALSE(veb.contains(7));

    // Duplicate insertion
    veb.insert(5);
    EXPECT_TRUE(veb.contains(5));

    // Predecessor & Successor tests
    EXPECT_EQ(veb.successor(1), 2);
    EXPECT_EQ(veb.successor(2), 5);
    EXPECT_EQ(veb.successor(5), 8);
    EXPECT_EQ(veb.successor(8), 13);
    EXPECT_EQ(veb.successor(13), std::nullopt);

    EXPECT_EQ(veb.predecessor(14), 13);
    EXPECT_EQ(veb.predecessor(13), 8);
    EXPECT_EQ(veb.predecessor(8), 5);
    EXPECT_EQ(veb.predecessor(5), 2);
    EXPECT_EQ(veb.predecessor(2), std::nullopt);

    // Erase min (2) -> new min should become 5
    veb.erase(2);
    EXPECT_FALSE(veb.contains(2));
    EXPECT_EQ(veb.min(), 5);
    EXPECT_EQ(veb.max(), 13);

    // Erase max (13) -> new max should become 8
    veb.erase(13);
    EXPECT_FALSE(veb.contains(13));
    EXPECT_EQ(veb.min(), 5);
    EXPECT_EQ(veb.max(), 8);

    // Erase remaining
    veb.erase(5);
    EXPECT_EQ(veb.min(), 8);
    EXPECT_EQ(veb.max(), 8);

    veb.erase(8);
    EXPECT_TRUE(veb.empty());
    EXPECT_EQ(veb.min(), std::nullopt);
    EXPECT_EQ(veb.max(), std::nullopt);
}

TEST(VEBTreeTest, OddBitWidths) {
    // Bits = 3 (upper_bits = 2, lower_bits = 1)
    VanEmdeBoasTree<3> veb3;
    veb3.insert(1);
    veb3.insert(3);
    veb3.insert(6);
    EXPECT_EQ(veb3.min(), 1);
    EXPECT_EQ(veb3.max(), 6);
    EXPECT_EQ(veb3.successor(1), 3);
    EXPECT_EQ(veb3.successor(3), 6);
    EXPECT_EQ(veb3.predecessor(6), 3);
    EXPECT_EQ(veb3.predecessor(3), 1);

    // Bits = 5 (upper_bits = 3, lower_bits = 2)
    VanEmdeBoasTree<5> veb5;
    for (uint32_t i = 0; i < 32; i += 3) {
        veb5.insert(i);
    }
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(veb5.contains(i), (i % 3 == 0));
    }
}

TEST(VEBTreeTest, Sparse64BitUniverse) {
    VanEmdeBoasTree<64> veb64;
    EXPECT_TRUE(veb64.empty());

    const uint64_t k1 = 100ULL;
    const uint64_t k2 = 0x100000000ULL;        // 2^32
    const uint64_t k3 = 0x7FFFFFFFFFFFFFFFULL; // Large 63-bit int
    const uint64_t k4 = 0xFFFFFFFFFFFFFFFFULL; // Max 64-bit int (~0ULL)

    veb64.insert(k1);
    veb64.insert(k2);
    veb64.insert(k3);
    veb64.insert(k4);

    EXPECT_FALSE(veb64.empty());
    EXPECT_EQ(veb64.min(), k1);
    EXPECT_EQ(veb64.max(), k4);

    EXPECT_TRUE(veb64.contains(k1));
    EXPECT_TRUE(veb64.contains(k2));
    EXPECT_TRUE(veb64.contains(k3));
    EXPECT_TRUE(veb64.contains(k4));
    EXPECT_FALSE(veb64.contains(0));
    EXPECT_FALSE(veb64.contains(k2 - 1));

    EXPECT_EQ(veb64.successor(k1), k2);
    EXPECT_EQ(veb64.successor(k2), k3);
    EXPECT_EQ(veb64.successor(k3), k4);
    EXPECT_EQ(veb64.successor(k4), std::nullopt);

    EXPECT_EQ(veb64.predecessor(k4), k3);
    EXPECT_EQ(veb64.predecessor(k3), k2);
    EXPECT_EQ(veb64.predecessor(k2), k1);
    EXPECT_EQ(veb64.predecessor(k1), std::nullopt);

    veb64.erase(k2);
    EXPECT_FALSE(veb64.contains(k2));
    EXPECT_EQ(veb64.successor(k1), k3);
    EXPECT_EQ(veb64.predecessor(k3), k1);

    veb64.clear();
    EXPECT_TRUE(veb64.empty());
    EXPECT_FALSE(veb64.contains(k1));
}

TEST(VEBTreeTest, SingleElementTransitions) {
    VanEmdeBoasTree<16> veb;
    veb.insert(42);
    EXPECT_FALSE(veb.empty());
    EXPECT_EQ(veb.min(), 42);
    EXPECT_EQ(veb.max(), 42);
    EXPECT_EQ(veb.successor(41), 42);
    EXPECT_EQ(veb.successor(42), std::nullopt);
    EXPECT_EQ(veb.predecessor(43), 42);
    EXPECT_EQ(veb.predecessor(42), std::nullopt);

    veb.erase(42);
    EXPECT_TRUE(veb.empty());
    EXPECT_EQ(veb.min(), std::nullopt);
    EXPECT_EQ(veb.max(), std::nullopt);
}

TEST(VEBTreeTest, EraseNonExistentElementIsNoOp) {
    VanEmdeBoasTree<16> veb;
    veb.insert(10);
    veb.insert(20);
    veb.insert(30);

    veb.erase(15); // not in tree
    veb.erase(5);  // smaller than min
    veb.erase(50); // larger than max

    EXPECT_EQ(veb.min(), 10);
    EXPECT_EQ(veb.max(), 30);
    EXPECT_TRUE(veb.contains(10));
    EXPECT_TRUE(veb.contains(20));
    EXPECT_TRUE(veb.contains(30));
}

TEST(VEBTreeTest, CopyConstructionAndAssignment) {
    VanEmdeBoasTree<16> tree1;
    tree1.insert(42);
    tree1.insert(100);
    tree1.insert(500);

    // Copy construction
    VanEmdeBoasTree<16> tree2 = tree1;
    EXPECT_TRUE(tree2.contains(42));
    EXPECT_TRUE(tree2.contains(100));
    EXPECT_TRUE(tree2.contains(500));
    EXPECT_EQ(tree2.min(), 42);
    EXPECT_EQ(tree2.max(), 500);

    // Mutate tree2; verify tree1 is independent
    tree2.insert(999);
    tree2.erase(42);
    EXPECT_FALSE(tree1.contains(999));
    EXPECT_TRUE(tree1.contains(42));
    EXPECT_TRUE(tree2.contains(999));
    EXPECT_FALSE(tree2.contains(42));

    // Copy assignment
    VanEmdeBoasTree<16> tree3;
    tree3 = tree1;
    EXPECT_TRUE(tree3.contains(42));
    EXPECT_TRUE(tree3.contains(100));
    EXPECT_TRUE(tree3.contains(500));
    EXPECT_FALSE(tree3.contains(999));

    // 64-bit universe copy test
    VanEmdeBoasTree<64> big_tree1;
    big_tree1.insert(1000ULL);
    big_tree1.insert(0xFEEDFACECAFEULL);
    VanEmdeBoasTree<64> big_tree2 = big_tree1;
    EXPECT_TRUE(big_tree2.contains(1000ULL));
    EXPECT_TRUE(big_tree2.contains(0xFEEDFACECAFEULL));
    big_tree2.erase(1000ULL);
    EXPECT_TRUE(big_tree1.contains(1000ULL));
    EXPECT_FALSE(big_tree2.contains(1000ULL));
}

TEST(VEBTreeTest, StressDifferentialSet1MOperations) {
    VanEmdeBoasTree<16> veb;
    std::set<uint32_t> ref_set;

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> key_dist(0, 65535);
    std::uniform_int_distribution<int> op_dist(0, 5);

    constexpr std::size_t num_ops = 1000000;

    for (std::size_t i = 0; i < num_ops; ++i) {
        int op = op_dist(rng);
        uint32_t x = key_dist(rng);

        switch (op) {
        case 0: // insert
            veb.insert(x);
            ref_set.insert(x);
            break;
        case 1: // erase
            veb.erase(x);
            ref_set.erase(x);
            break;
        case 2: // contains
            ASSERT_EQ(veb.contains(x), (ref_set.count(x) > 0));
            break;
        case 3: { // successor
            auto it = ref_set.upper_bound(x);
            std::optional<uint32_t> expected =
                (it != ref_set.end()) ? std::make_optional(*it) : std::nullopt;
            ASSERT_EQ(veb.successor(x), expected);
            break;
        }
        case 4: { // predecessor
            auto it = ref_set.lower_bound(x);
            std::optional<uint32_t> expected = std::nullopt;
            if (it != ref_set.begin()) {
                --it;
                expected = *it;
            }
            ASSERT_EQ(veb.predecessor(x), expected);
            break;
        }
        case 5: { // min & max & empty
            ASSERT_EQ(veb.empty(), ref_set.empty());
            if (ref_set.empty()) {
                ASSERT_EQ(veb.min(), std::nullopt);
                ASSERT_EQ(veb.max(), std::nullopt);
            } else {
                ASSERT_EQ(veb.min(), *ref_set.begin());
                ASSERT_EQ(veb.max(), *ref_set.rbegin());
            }
            break;
        }
        }
    }
}
