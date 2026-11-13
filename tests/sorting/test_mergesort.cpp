#include "../core/pmr_test_helpers.hpp"
#include "algoat/core/scratch_buffer.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "test_stable_item.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <gtest/gtest.h>
#include <memory_resource>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

ALGOAT_INSTALL_GLOBAL_ALLOC_COUNTER()

using namespace algoat::sorting;
using namespace algoat::sorting::testing;

namespace {

using ::algoat::testing::CountingMemoryResource;
using ::algoat::testing::ScopedGlobalAllocTracker;

struct ItemWithKey {
    int key;
    int seq;
};

} // namespace

class MergeSortTest : public ::testing::Test {
protected:
    MergeSort algo;
};

TEST_F(MergeSortTest, IsStable) {
    test_sort_stability(algo);
}

TEST_F(MergeSortTest, ExactFitStackArenaDoesNotOverflow) {
    // 256 ints == 1024 bytes; StackArena<1024> has 64-byte alignment and no extra padding waste.
    std::vector<int> data(256);
    for (int i = 0; i < 256; ++i) {
        data[i] = 256 - i;
    }
    algoat::core::StackArena<1024> arena;
    mergesort(std::span{data}, arena.resource());
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST_F(MergeSortTest, ThrowsBadAllocWhenStackArenaInsufficient) {
    std::vector<int> data(256, 1);
    data[0] = 9;
    algoat::core::StackArena<64> small_arena;
    EXPECT_THROW(mergesort(std::span{data}, small_arena.resource()), std::bad_alloc);
}

TEST_F(MergeSortTest, RangeAndAdapterPmrOverloadsWork) {
    algoat::core::StackArena<65536> arena;

    std::vector<int> v1 = {5, 1, 9, 3, 7, 2, 8, 4, 6};
    mergesort(v1, arena.resource());
    EXPECT_TRUE(std::is_sorted(v1.begin(), v1.end()));

    std::vector<ItemWithKey> v2 = {{3, 0}, {1, 1}, {3, 2}, {2, 3}};
    mergesort(v2, std::ranges::greater{}, &ItemWithKey::key, arena.resource());
    EXPECT_EQ(v2[0].key, 3);
    EXPECT_EQ(v2[0].seq, 0);
    EXPECT_EQ(v2[1].key, 3);
    EXPECT_EQ(v2[1].seq, 2);
    EXPECT_EQ(v2[2].key, 2);
    EXPECT_EQ(v2[3].key, 1);

    std::vector<int> v3 = {4, 2, 5, 1, 3};
    algo.sort(std::span{v3}, arena.resource());
    EXPECT_TRUE(std::is_sorted(v3.begin(), v3.end()));

    std::vector<int> v4 = {4, 2, 5, 1, 3};
    algo.sort(std::span{v4}, std::greater<>{}, arena.resource());
    EXPECT_TRUE(std::is_sorted(v4.begin(), v4.end(), std::greater<>{}));
}

TEST_F(MergeSortTest, NonTrivialStringsSortWithStackArenaAndStability) {
    std::vector<std::string> words = {"delta", "alpha", "charlie", "bravo", "alpha"};
    algoat::core::StackArena<65536> arena;
    mergesort(std::span{words}, arena.resource());
    EXPECT_EQ(words, (std::vector<std::string>{"alpha", "alpha", "bravo", "charlie", "delta"}));

    std::vector<StableItem> items;
    for (int i = 0; i < 200; ++i) {
        items.push_back({(i * 37) % 13, i});
    }
    mergesort(std::span{items}, arena.resource());
    verify_stability(items);
}

TEST_F(MergeSortTest, CountingResourceReceivesExactSingleAllocation) {
    std::vector<int> data = {9, 4, 7, 1, 8, 2, 6, 3, 5};
    CountingMemoryResource mr;
    mergesort(std::span{data}, &mr);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(mr.allocate_calls(), 1U);
    EXPECT_EQ(mr.deallocate_calls(), 1U);
}

TEST_F(MergeSortTest, ZeroGlobalHeapAllocationsWithStackArena) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> data(4096);
    for (int& x : data) {
        x = dist(gen);
    }
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    ScopedGlobalAllocTracker tracker;
    {
        algoat::core::StackArena<65536> arena;
        mergesort(std::span{data}, arena.resource());
    }
    EXPECT_EQ(tracker.count(), 0U);
    EXPECT_EQ(data, expected);
}

TEST_F(MergeSortTest, NonContiguousRangeUsesPmrWithoutHeapAllocations) {
    std::deque<StableItem> dq;
    for (int i = 0; i < 300; ++i) {
        dq.push_back({(i * 41) % 17, i});
    }

    ScopedGlobalAllocTracker tracker;
    {
        algoat::core::StackArena<65536> arena;
        mergesort(dq, arena.resource());
    }
    EXPECT_EQ(tracker.count(), 0U);

    for (std::size_t i = 1; i < dq.size(); ++i) {
        EXPECT_LE(dq[i - 1].key, dq[i].key);
        if (dq[i - 1].key == dq[i].key) {
            EXPECT_LT(dq[i - 1].original_index, dq[i].original_index);
        }
    }
}

TEST_F(MergeSortTest, StaticExtentSpanSupportsPmrOverloads) {
    std::array<int, 8> a1 = {8, 3, 7, 1, 6, 2, 5, 4};
    std::array<int, 8> a2 = a1;
    std::array<int, 8> a3 = a1;
    algoat::core::StackArena<1024> arena;

    mergesort(std::span<int, 8>{a1}, arena.resource());
    EXPECT_TRUE(std::is_sorted(a1.begin(), a1.end()));

    arena.release();
    mergesort(std::span<int, 8>{a2}, std::greater<>{}, arena.resource());
    EXPECT_TRUE(std::is_sorted(a2.begin(), a2.end(), std::greater<>{}));

    arena.release();
    MergeSort{}.sort(std::span<int, 8>{a3}, arena.resource());
    EXPECT_TRUE(std::is_sorted(a3.begin(), a3.end()));
}
