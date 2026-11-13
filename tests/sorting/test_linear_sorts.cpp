#include "../core/pmr_test_helpers.hpp"
#include "algoat/core/scratch_buffer.hpp"
#include "algoat/sorting/bucketsort.hpp"
#include "algoat/sorting/countingsort.hpp"
#include "algoat/sorting/pigeonholesort.hpp"
#include "algoat/sorting/radixsort.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory_resource>
#include <new>
#include <random>
#include <vector>

ALGOAT_INSTALL_GLOBAL_ALLOC_COUNTER()

using namespace algoat::sorting;

template <typename Algo> class LinearSortTest : public ::testing::Test {
protected:
    Algo algo;
    void verify_sort(std::vector<int>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

using LinearAlgos = ::testing::Types<CountingSort, PigeonholeSort, RadixSortLSD, RadixSortMSD,
                                     RadixSortInPlaceMSD, BucketSort>;
TYPED_TEST_SUITE(LinearSortTest, LinearAlgos);

TYPED_TEST(LinearSortTest, RandomData) {
    std::vector<int> data = {5, -3, 8, 1, -9, 2, 7};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, EmptyInput) {
    std::vector<int> data;
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, SingleElement) {
    std::vector<int> data = {42};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, AllDuplicates) {
    std::vector<int> data = {7, 7, 7, 7, 7};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, AlreadySorted) {
    std::vector<int> data = {-5, -2, 0, 4, 5, 6, 7};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, ReverseSorted) {
    std::vector<int> data = {7, 6, 5, 4, 0, -2, -5};
    this->verify_sort(data);
}

TYPED_TEST(LinearSortTest, LargeRandom) {
    std::vector<int> data(1000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (int& x : data) {
        x = dist(gen);
    }
    this->verify_sort(data);
}

TEST(RadixSortTest, SignedIntegerBoundaries) {
    std::vector<int32_t> data = {
        std::numeric_limits<int32_t>::max(),     std::numeric_limits<int32_t>::min(),    0, -1, 1,
        std::numeric_limits<int32_t>::min() + 1, std::numeric_limits<int32_t>::max() - 1};
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::sorting::RadixSortLSD{}.sort(std::span{data});
    EXPECT_EQ(data, expected);
}

template <typename Algo> class RangeSortTest : public ::testing::Test {
protected:
    Algo algo;
    template <typename T> void verify_sort(std::vector<T>& data) {
        auto expected = data;
        std::sort(expected.begin(), expected.end());
        algo.sort(std::span{data});
        EXPECT_EQ(data, expected);
    }
};

using RangeAlgos = ::testing::Types<CountingSort, PigeonholeSort, BucketSort>;
TYPED_TEST_SUITE(RangeSortTest, RangeAlgos);

TYPED_TEST(RangeSortTest, FullInt8Range) {
    std::vector<int8_t> data;
    for (int i = -128; i <= 127; ++i) {
        data.push_back(static_cast<int8_t>(i));
    }
    std::reverse(data.begin(), data.end());
    this->verify_sort(data);
}

TYPED_TEST(RangeSortTest, ClusteredNearMinAndMax) {
    constexpr int32_t min_v = std::numeric_limits<int32_t>::min();
    std::vector<int32_t> min_cluster = {min_v + 10, min_v, min_v + 5, min_v + 2, min_v};
    this->verify_sort(min_cluster);

    constexpr int32_t max_v = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> max_cluster = {max_v - 5, max_v, max_v - 10, max_v - 2, max_v};
    this->verify_sort(max_cluster);
}

template <typename Algo, typename T>
concept CanSort = requires(Algo a, std::span<T> arr) { a.sort(arr); };

TEST(AlgorithmConstraintTest, RejectsBoolAtCompileTime) {
    static_assert(!CanSort<algoat::sorting::CountingSort, bool>, "CountingSort must reject bool");
    static_assert(!CanSort<algoat::sorting::BucketSort, bool>, "BucketSort must reject bool");
    static_assert(!CanSort<algoat::sorting::PigeonholeSort, bool>,
                  "PigeonholeSort must reject bool");
    static_assert(!CanSort<algoat::sorting::RadixSortLSD, bool>, "RadixSortLSD must reject bool");
    static_assert(!CanSort<algoat::sorting::RadixSortMSD, bool>, "RadixSortMSD must reject bool");
    static_assert(!CanSort<algoat::sorting::RadixSortInPlaceMSD, bool>,
                  "RadixSortInPlaceMSD must reject bool");
    SUCCEED();
}

namespace {

using ::algoat::testing::CountingMemoryResource;
using ::algoat::testing::ScopedGlobalAllocTracker;

template <typename T> void verify_radix_pmr_all_variants(std::vector<T> seed_data) {
    auto expected = seed_data;
    std::sort(expected.begin(), expected.end());

    algoat::core::StackArena<65536> arena;

    auto d1 = seed_data;
    radixsort_lsd(std::span{d1}, arena.resource());
    EXPECT_EQ(d1, expected);

    arena.release();
    auto d2 = seed_data;
    radixsort_msd(std::span{d2}, arena.resource());
    EXPECT_EQ(d2, expected);

    arena.release();
    auto d3 = seed_data;
    radixsort_lsd(d3, arena.resource());
    EXPECT_EQ(d3, expected);

    arena.release();
    auto d4 = seed_data;
    radixsort_msd(d4, arena.resource());
    EXPECT_EQ(d4, expected);

    arena.release();
    auto d5 = seed_data;
    RadixSortLSD{}.sort(std::span{d5}, arena.resource());
    EXPECT_EQ(d5, expected);

    arena.release();
    auto d6 = seed_data;
    RadixSortMSD{}.sort(std::span{d6}, arena.resource());
    EXPECT_EQ(d6, expected);

    arena.release();
    auto d7 = seed_data;
    radixsort(d7, arena.resource());
    EXPECT_EQ(d7, expected);
}

} // namespace

TEST(RadixSortPmrTest, SortsMultiWidthIntegralTypesWithStackArena) {
    verify_radix_pmr_all_variants<std::int8_t>({127, -128, 0, -1, 42, -42, 17, -17});
    verify_radix_pmr_all_variants<std::uint16_t>({65535, 0, 1024, 512, 255, 1, 32768});
    verify_radix_pmr_all_variants<std::int32_t>({std::numeric_limits<std::int32_t>::max(),
                                                 std::numeric_limits<std::int32_t>::min(), 0, -5, 5,
                                                 -1000, 1000});
    verify_radix_pmr_all_variants<std::uint64_t>(
        {std::numeric_limits<std::uint64_t>::max(), 0ULL, 1ULL << 63, (1ULL << 63) - 1, 999ULL});
}

TEST(RadixSortPmrTest, ExactFitAndOverflowBehavior) {
    std::vector<std::int32_t> data(256);
    for (int i = 0; i < 256; ++i) {
        data[i] = 255 - i;
    }
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::core::StackArena<1024> exact_arena;
    auto lsd_data = data;
    radixsort_lsd(std::span{lsd_data}, exact_arena.resource());
    EXPECT_EQ(lsd_data, expected);

    exact_arena.release();
    auto msd_data = data;
    radixsort_msd(std::span{msd_data}, exact_arena.resource());
    EXPECT_EQ(msd_data, expected);

    algoat::core::StackArena<64> tiny_arena;
    EXPECT_THROW(radixsort_lsd(std::span{data}, tiny_arena.resource()), std::bad_alloc);
    EXPECT_THROW(radixsort_msd(std::span{data}, tiny_arena.resource()), std::bad_alloc);
}

TEST(RadixSortPmrTest, CountingResourceAllocationCounts) {
    CountingMemoryResource mr;
    std::vector<int> single = {42};
    radixsort_lsd(std::span{single}, &mr);
    radixsort_msd(std::span{single}, &mr);
    EXPECT_EQ(mr.allocate_calls(), 0U);
    EXPECT_EQ(mr.deallocate_calls(), 0U);

    std::vector<int> data = {9, -3, 7, 1, -5, 4};
    radixsort_lsd(std::span{data}, &mr);
    EXPECT_EQ(mr.allocate_calls(), 1U);
    EXPECT_EQ(mr.deallocate_calls(), 1U);

    std::vector<int> data2 = {9, -3, 7, 1, -5, 4};
    radixsort_msd(std::span{data2}, &mr);
    EXPECT_EQ(mr.allocate_calls(), 2U);
    EXPECT_EQ(mr.deallocate_calls(), 2U);
}

TEST(RadixSortPmrTest, ZeroGlobalHeapAllocationsWithStackArena) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<std::int32_t> dist(-100000, 100000);
    std::vector<std::int32_t> base(4096);
    for (auto& x : base) {
        x = dist(gen);
    }
    auto expected = base;
    std::sort(expected.begin(), expected.end());

    auto lsd_data = base;
    auto msd_data = base;

    ScopedGlobalAllocTracker tracker;
    {
        algoat::core::StackArena<65536> arena;
        radixsort_lsd(std::span{lsd_data}, arena.resource());
        arena.release();
        radixsort_msd(std::span{msd_data}, arena.resource());
    }
    EXPECT_EQ(tracker.count(), 0U);
    EXPECT_EQ(lsd_data, expected);
    EXPECT_EQ(msd_data, expected);
}

TEST(RadixSortPmrTest, StaticExtentSpanSupportsPmrOverloads) {
    std::array<int, 8> a1 = {40, -10, 7, 0, 99, -5, 12, 3};
    auto expected = a1;
    std::sort(expected.begin(), expected.end());

    auto a2 = a1;
    auto a3 = a1;
    auto a4 = a1;
    auto a5 = a1;
    algoat::core::StackArena<1024> arena;

    radixsort(std::span<int, 8>{a1}, arena.resource());
    EXPECT_EQ(a1, expected);

    arena.release();
    radixsort_lsd(std::span<int, 8>{a2}, arena.resource());
    EXPECT_EQ(a2, expected);

    arena.release();
    radixsort_msd(std::span<int, 8>{a3}, arena.resource());
    EXPECT_EQ(a3, expected);

    arena.release();
    RadixSortLSD{}.sort(std::span<int, 8>{a4}, arena.resource());
    EXPECT_EQ(a4, expected);

    arena.release();
    RadixSortMSD{}.sort(std::span<int, 8>{a5}, arena.resource());
    EXPECT_EQ(a5, expected);
}
