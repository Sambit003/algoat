/**
 * @file test_simd_sorting_network.cpp
 * @brief Unit and verification tests for in-register SIMD sorting networks (N <= 64).
 */

#include "algoat/meta/simd_sorting_network.hpp"
#include "algoat/meta/static_sort.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace {

template <std::size_t N, typename T>
void verify_random_trials(std::size_t trials, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (std::size_t t = 0; t < trials; ++t) {
        std::array<T, N> actual{};
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(static_cast<T>(-1e6), static_cast<T>(1e6));
            for (auto& x : actual) {
                x = dist(rng);
            }
        } else {
            using DistT = std::conditional_t<(sizeof(T) < sizeof(int)), int, T>;
            std::uniform_int_distribution<DistT> dist(
                static_cast<DistT>(std::numeric_limits<T>::min()),
                static_cast<DistT>(std::numeric_limits<T>::max()));
            for (auto& x : actual) {
                x = static_cast<T>(dist(rng));
            }
        }
        auto expected = actual;
        std::sort(expected.begin(), expected.end());

        algoat::meta::sort_simd_network(std::span<T, N>{actual});
        ASSERT_EQ(actual, expected) << "Mismatch at trial " << t << " for N=" << N;
    }
}

template <std::size_t N> void verify_zero_one_exhaustive() {
    for (std::uint32_t mask = 0; mask < (1u << N); ++mask) {
        std::array<std::int32_t, N> actual{};
        for (std::size_t i = 0; i < N; ++i) {
            actual[i] = static_cast<std::int32_t>((mask >> i) & 1u);
        }
        auto expected = actual;
        std::sort(expected.begin(), expected.end());

        algoat::meta::SIMDSortingNetwork<N, std::int32_t>::sort(std::span<std::int32_t, N>{actual});
        ASSERT_EQ(actual, expected) << "Failed for N=" << N << " 0-1 mask=" << mask;
    }
}

template <typename T> void verify_fp_edge_cases() {
    std::array<T, 16> data = {
        std::numeric_limits<T>::infinity(),
        static_cast<T>(-0.0),
        static_cast<T>(+0.0),
        -std::numeric_limits<T>::infinity(),
        std::numeric_limits<T>::denorm_min(),
        -std::numeric_limits<T>::denorm_min(),
        static_cast<T>(3.141592653589793),
        static_cast<T>(-2.718281828459045),
        static_cast<T>(-0.0),
        static_cast<T>(+0.0),
        std::numeric_limits<T>::max(),
        std::numeric_limits<T>::lowest(),
        static_cast<T>(1.0),
        static_cast<T>(-1.0),
        static_cast<T>(0.5),
        static_cast<T>(-0.5),
    };

    auto count_signed_zeros = [](const std::array<T, 16>& arr) {
        std::pair<std::size_t, std::size_t> counts{0, 0};
        for (T v : arr) {
            if (v == static_cast<T>(0.0)) {
                if (std::signbit(v)) {
                    ++counts.first;
                } else {
                    ++counts.second;
                }
            }
        }
        return counts;
    };

    const auto zeros_before = count_signed_zeros(data);
    auto expected = data;
    std::sort(expected.begin(), expected.end());

    algoat::meta::sort_simd_network(std::span<T, 16>{data});
    EXPECT_EQ(data, expected);
    EXPECT_EQ(count_signed_zeros(data), zeros_before);
}

TEST(SimdSortingNetworkTest, SortsEightInt32ZeroOneExhaustive) {
    verify_zero_one_exhaustive<8>();
}

TEST(SimdSortingNetworkTest, SortsSixteenInt32ZeroOneExhaustive) {
    verify_zero_one_exhaustive<16>();
}

TEST(SimdSortingNetworkTest, SortsAll32BitTypesAndSizes) {
    verify_random_trials<8, std::int32_t>(5000, 101);
    verify_random_trials<16, std::int32_t>(5000, 102);
    verify_random_trials<32, std::int32_t>(5000, 103);
    verify_random_trials<64, std::int32_t>(5000, 104);

    verify_random_trials<8, std::uint32_t>(5000, 201);
    verify_random_trials<16, std::uint32_t>(5000, 202);
    verify_random_trials<32, std::uint32_t>(5000, 203);
    verify_random_trials<64, std::uint32_t>(5000, 204);

    verify_random_trials<8, float>(5000, 301);
    verify_random_trials<16, float>(5000, 302);
    verify_random_trials<32, float>(5000, 303);
    verify_random_trials<64, float>(5000, 304);
}

TEST(SimdSortingNetworkTest, FloatEdgeCasesInfinitiesAndSignedZero) {
    verify_fp_edge_cases<float>();
}

TEST(SimdSortingNetworkTest, SortsAll64BitTypesAndSizes) {
    verify_random_trials<8, std::int64_t>(5000, 401);
    verify_random_trials<16, std::int64_t>(5000, 402);
    verify_random_trials<32, std::int64_t>(5000, 403);
    verify_random_trials<64, std::int64_t>(5000, 404);

    verify_random_trials<8, std::uint64_t>(5000, 501);
    verify_random_trials<16, std::uint64_t>(5000, 502);
    verify_random_trials<32, std::uint64_t>(5000, 503);
    verify_random_trials<64, std::uint64_t>(5000, 504);

    verify_random_trials<8, double>(5000, 601);
    verify_random_trials<16, double>(5000, 602);
    verify_random_trials<32, double>(5000, 603);
    verify_random_trials<64, double>(5000, 604);

    verify_random_trials<16, long long>(1000, 701);
    verify_random_trials<16, unsigned long long>(1000, 702);
}

TEST(SimdSortingNetworkTest, Int64AndUint64ExtremeBoundaries) {
    std::array<std::int64_t, 8> signed_data = {
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::min(),
        0,
        -1,
        1,
        std::numeric_limits<std::int64_t>::min() + 1,
        std::numeric_limits<std::int64_t>::max() - 1,
        -42,
    };
    auto signed_expected = signed_data;
    std::sort(signed_expected.begin(), signed_expected.end());
    algoat::meta::sort_simd_network(std::span<std::int64_t, 8>{signed_data});
    EXPECT_EQ(signed_data, signed_expected);

    std::array<std::uint64_t, 16> unsigned_data = {
        std::numeric_limits<std::uint64_t>::max(),
        0ULL,
        1ULL,
        1ULL << 63,
        (1ULL << 63) - 1ULL,
        (1ULL << 63) + 1ULL,
        std::numeric_limits<std::uint64_t>::max() - 1ULL,
        0x7FFFFFFFFFFFFFFFULL,
        0x8000000000000000ULL,
        0xFFFFFFFF00000000ULL,
        0x00000000FFFFFFFFULL,
        42ULL,
        17ULL,
        (1ULL << 32),
        (1ULL << 31),
        (1ULL << 62),
    };
    auto unsigned_expected = unsigned_data;
    std::sort(unsigned_expected.begin(), unsigned_expected.end());
    algoat::meta::sort_simd_network(std::span<std::uint64_t, 16>{unsigned_data});
    EXPECT_EQ(unsigned_data, unsigned_expected);
}

TEST(SimdSortingNetworkTest, DoubleEdgeCasesInfinitiesAndSignedZero) {
    verify_fp_edge_cases<double>();
}

TEST(SimdSortingNetworkTest, ScalarFallbackNonSimdArithmeticTypes) {
    verify_random_trials<8, std::int16_t>(500, 801);
    verify_random_trials<16, std::uint16_t>(500, 802);
    verify_random_trials<32, std::int8_t>(500, 803);
    verify_random_trials<64, std::uint8_t>(500, 804);
}

template <std::size_t N, typename T>
void verify_fast_permutations(std::size_t count, std::uint64_t& state) {
    auto next_u64 = [&]() noexcept -> std::uint64_t {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };

    for (std::size_t t = 0; t < count; ++t) {
        std::array<T, N> actual{};
        for (std::size_t i = 0; i < N; ++i) {
            std::uint64_t r = next_u64();
            if constexpr (std::is_floating_point_v<T>) {
                // Bounded finite values avoiding NaNs
                actual[i] = static_cast<T>(static_cast<std::int32_t>(r & 0xFFFFFFFFULL)) *
                            static_cast<T>(0.0009765625);
            } else {
                actual[i] = static_cast<T>(r);
            }
        }
        auto expected = actual;
        std::sort(expected.begin(), expected.end());

        algoat::meta::sort_simd_network(std::span<T, N>{actual});
        ASSERT_EQ(actual, expected) << "Permutation mismatch at N=" << N << " trial=" << t;
    }
}

TEST(SimdSortingNetworkTest, EquivalenceWithStdSortOneMillionPermutations) {
    // 24 (N, T) combinations * 42,000 trials = 1,008,000 random permutations >= 1,000,000.
    constexpr std::size_t kPerSlice = 42000;
    std::uint64_t state = 0xdeadbeefcafebabeULL;

    verify_fast_permutations<8, std::int32_t>(kPerSlice, state);
    verify_fast_permutations<16, std::int32_t>(kPerSlice, state);
    verify_fast_permutations<32, std::int32_t>(kPerSlice, state);
    verify_fast_permutations<64, std::int32_t>(kPerSlice, state);

    verify_fast_permutations<8, std::uint32_t>(kPerSlice, state);
    verify_fast_permutations<16, std::uint32_t>(kPerSlice, state);
    verify_fast_permutations<32, std::uint32_t>(kPerSlice, state);
    verify_fast_permutations<64, std::uint32_t>(kPerSlice, state);

    verify_fast_permutations<8, float>(kPerSlice, state);
    verify_fast_permutations<16, float>(kPerSlice, state);
    verify_fast_permutations<32, float>(kPerSlice, state);
    verify_fast_permutations<64, float>(kPerSlice, state);

    verify_fast_permutations<8, std::int64_t>(kPerSlice, state);
    verify_fast_permutations<16, std::int64_t>(kPerSlice, state);
    verify_fast_permutations<32, std::int64_t>(kPerSlice, state);
    verify_fast_permutations<64, std::int64_t>(kPerSlice, state);

    verify_fast_permutations<8, std::uint64_t>(kPerSlice, state);
    verify_fast_permutations<16, std::uint64_t>(kPerSlice, state);
    verify_fast_permutations<32, std::uint64_t>(kPerSlice, state);
    verify_fast_permutations<64, std::uint64_t>(kPerSlice, state);

    verify_fast_permutations<8, double>(kPerSlice, state);
    verify_fast_permutations<16, double>(kPerSlice, state);
    verify_fast_permutations<32, double>(kPerSlice, state);
    verify_fast_permutations<64, double>(kPerSlice, state);
}

consteval std::array<std::int32_t, 8> constexpr_static_sort_8() {
    std::array<std::int32_t, 8> arr = {8, 3, 7, 1, 5, 2, 6, 4};
    algoat::meta::static_sort(std::span<std::int32_t, 8>{arr});
    return arr;
}

TEST(SimdSortingNetworkTest, StaticSortIntegrationAndConstexprFallback) {
    constexpr auto sorted8 = constexpr_static_sort_8();
    static_assert(sorted8 == std::array<std::int32_t, 8>{1, 2, 3, 4, 5, 6, 7, 8});

    std::array<std::int32_t, 32> d32{};
    for (std::size_t i = 0; i < 32; ++i) {
        d32[i] = static_cast<std::int32_t>(32 - i);
    }
    algoat::meta::static_sort(std::span<std::int32_t, 32>{d32});
    EXPECT_TRUE(std::is_sorted(d32.begin(), d32.end()));

    std::array<double, 64> d64{};
    for (std::size_t i = 0; i < 64; ++i) {
        d64[i] = static_cast<double>(64 - i);
    }
    algoat::meta::static_sort(std::span<double, 64>{d64});
    EXPECT_TRUE(std::is_sorted(d64.begin(), d64.end()));

    // Custom comparator still works via scalar fallback
    std::array<std::int32_t, 8> desc = {1, 2, 3, 4, 5, 6, 7, 8};
    algoat::meta::static_sort<algoat::meta::PolicyDescriptor{}>(std::span<std::int32_t, 8>{desc},
                                                                std::greater<>{});
    EXPECT_TRUE(std::is_sorted(desc.begin(), desc.end(), std::greater<>{}));
}

} // namespace
