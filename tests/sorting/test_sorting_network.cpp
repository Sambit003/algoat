/**
 * @file test_sorting_network.cpp
 * @brief Unit tests for zero-overhead compile-time sorting networks.
 */

#include "algoat/meta/sorting_network.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

// Compile-time constexpr validation of sorting networks
constexpr bool test_constexpr_sorting_network() {
    std::array<int, 4> a4 = {4, 1, 3, 2};
    algoat::meta::sort_network<int, 4>(std::span<int, 4>{a4});
    if (a4[0] != 1 || a4[1] != 2 || a4[2] != 3 || a4[3] != 4) {
        return false;
    }

    std::array<int, 8> a8 = {8, 7, 6, 5, 4, 3, 2, 1};
    algoat::meta::sort_network<int, 8>(std::span<int, 8>{a8});
    for (int i = 0; i < 8; ++i) {
        if (a8[static_cast<std::size_t>(i)] != i + 1) {
            return false;
        }
    }

    std::array<int, 16> a16 = {16, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8};
    algoat::meta::sort_network<int, 16>(std::span<int, 16>{a16});
    for (int i = 0; i < 16; ++i) {
        if (a16[static_cast<std::size_t>(i)] != i + 1) {
            return false;
        }
    }

    return true;
}

static_assert(test_constexpr_sorting_network(),
              "Sorting network must execute and sort correctly at compile time.");

} // namespace

TEST(SortingNetworkTest, HandlesTrivialSizes) {
    std::array<int, 0> a0{};
    algoat::meta::sort_network<int, 0>(std::span<int, 0>{a0});

    std::array<int, 1> a1{42};
    algoat::meta::sort_network<int, 1>(std::span<int, 1>{a1});
    EXPECT_EQ(a1[0], 42);
}

TEST(SortingNetworkTest, SortsAllSizesFromTwoToSixteen) {
    auto test_size = []<std::size_t N>() {
        std::array<int, N> arr;
        std::iota(arr.begin(), arr.end(), 1);

        // Test reversed
        std::reverse(arr.begin(), arr.end());
        algoat::meta::sort_network<int, N>(std::span<int, N>{arr});
        EXPECT_TRUE(std::is_sorted(arr.begin(), arr.end())) << "Failed on reversed size N=" << N;

        // Test with random permutations
        std::mt19937 rng(42 + N);
        for (int trial = 0; trial < 20; ++trial) {
            std::shuffle(arr.begin(), arr.end(), rng);
            algoat::meta::sort_network<int, N>(std::span<int, N>{arr});
            EXPECT_TRUE(std::is_sorted(arr.begin(), arr.end()))
                << "Failed on random trial " << trial << " for size N=" << N;
        }

        // Test with duplicates
        for (std::size_t i = 0; i < N; ++i) {
            arr[i] = static_cast<int>(i % 3);
        }
        std::shuffle(arr.begin(), arr.end(), rng);
        algoat::meta::sort_network<int, N>(std::span<int, N>{arr});
        EXPECT_TRUE(std::is_sorted(arr.begin(), arr.end()))
            << "Failed on duplicates for size N=" << N;
    };

    test_size.template operator()<2>();
    test_size.template operator()<3>();
    test_size.template operator()<4>();
    test_size.template operator()<5>();
    test_size.template operator()<6>();
    test_size.template operator()<7>();
    test_size.template operator()<8>();
    test_size.template operator()<9>();
    test_size.template operator()<10>();
    test_size.template operator()<11>();
    test_size.template operator()<12>();
    test_size.template operator()<13>();
    test_size.template operator()<14>();
    test_size.template operator()<15>();
    test_size.template operator()<16>();
}

TEST(SortingNetworkTest, SupportsCustomComparator) {
    std::array<int, 8> data = {1, 5, 2, 8, 3, 7, 4, 6};
    algoat::meta::sort_network<int, 8>(std::span<int, 8>{data}, std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
}
