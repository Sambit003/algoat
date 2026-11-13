#include "algoat/core/hardware_topology.hpp"
#include "algoat/simd/isa.hpp"

#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__unix__)
#include <unistd.h>
#endif

namespace {

using algoat::core::CacheProperties;
using algoat::core::HardwareTopology;

TEST(HardwareTopologyTest, SingletonAddressStabilityAndThreadSafety) {
    const HardwareTopology* first_ptr = &HardwareTopology::instance();
    EXPECT_EQ(first_ptr, &HardwareTopology::instance());

    constexpr int kThreads = 8;
    std::vector<const HardwareTopology*> ptrs(kThreads, nullptr);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&ptrs, i]() { ptrs[i] = &HardwareTopology::instance(); });
    }
    for (auto& t : threads) {
        t.join();
    }
    for (const auto* p : ptrs) {
        EXPECT_EQ(p, first_ptr);
    }
}

TEST(HardwareTopologyTest, ReportsValidCacheMetricsAndProperties) {
    const auto& topo = HardwareTopology::instance();

    EXPECT_GT(topo.l1d_cache_size(), 0U);
    EXPECT_GT(topo.l2_cache_size(), 0U);
    EXPECT_GE(topo.l2_cache_size(), topo.l1d_cache_size());

    const std::size_t line_size = topo.cache_line_size();
    EXPECT_TRUE(std::has_single_bit(line_size));
    EXPECT_TRUE(line_size == 64U || line_size == 128U || line_size == 32U);

    const CacheProperties& l1d = topo.l1d_properties();
    EXPECT_EQ(l1d.level, 1U);
    EXPECT_EQ(l1d.size_bytes, topo.l1d_cache_size());
    EXPECT_EQ(l1d.line_size_bytes, line_size);
}

TEST(HardwareTopologyTest, VectorExtensionsConsistentWithHostIsa) {
    const auto& topo = HardwareTopology::instance();
    const auto host_isa = algoat::simd::detect_host_isa();

    if (host_isa == algoat::simd::InstructionSet::AVX512) {
        EXPECT_TRUE(topo.has_avx512f());
        EXPECT_TRUE(topo.has_avx2());
    } else if (host_isa == algoat::simd::InstructionSet::AVX2) {
        EXPECT_TRUE(topo.has_avx2());
    } else if (host_isa == algoat::simd::InstructionSet::Neon) {
        EXPECT_TRUE(topo.has_neon());
    }

#if defined(__x86_64__) || defined(_M_X64)
    EXPECT_FALSE(topo.has_neon());
    if (topo.has_avx512f()) {
        EXPECT_TRUE(topo.has_avx2());
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_TRUE(topo.has_neon());
    EXPECT_FALSE(topo.has_avx2());
    EXPECT_FALSE(topo.has_avx512f());
    EXPECT_FALSE(topo.has_bmi2());
#endif
}

TEST(HardwareTopologyTest, CacheMetricsMatchOsSysconfWhenAvailable) {
    const auto& topo = HardwareTopology::instance();

#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
    const long os_l1d = ::sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (os_l1d > 0) {
        EXPECT_EQ(topo.l1d_cache_size(), static_cast<std::size_t>(os_l1d));
    }
#endif
#if defined(__linux__) && defined(_SC_LEVEL1_DCACHE_LINESIZE)
    const long os_line = ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (os_line > 0) {
        EXPECT_EQ(topo.cache_line_size(), static_cast<std::size_t>(os_line));
    }
#endif
#if defined(__linux__) && defined(_SC_LEVEL2_CACHE_SIZE)
    const long os_l2 = ::sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (os_l2 > 0) {
        EXPECT_EQ(topo.l2_cache_size(), static_cast<std::size_t>(os_l2));
    }
#endif
#if defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
    const long os_l3 = ::sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (os_l3 > 0) {
        EXPECT_EQ(topo.l3_cache_size(), static_cast<std::size_t>(os_l3));
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    EXPECT_GT(topo.l1d_properties().associativity, 0U);
#endif
}

TEST(HardwareTopologyTest, OptimalL1BlockElementsComputesAccurateAndSafeBounds) {
    const auto& topo = HardwareTopology::instance();
    const std::size_t l1d = topo.l1d_cache_size();

    EXPECT_EQ(topo.optimal_l1_block_elements<std::int32_t>(),
              static_cast<std::size_t>((l1d * 0.5) / sizeof(std::int32_t)));
    EXPECT_EQ(topo.optimal_l1_block_elements<double>(0.25),
              static_cast<std::size_t>((l1d * 0.25) / sizeof(double)));
    EXPECT_EQ(topo.optimal_l1_block_elements<std::uint8_t>(1.0), l1d);

    struct alignas(64) CacheLineBlock {
        std::uint8_t bytes[64];
    };
    EXPECT_EQ(topo.optimal_l1_block_elements<CacheLineBlock>(0.5),
              static_cast<std::size_t>((l1d * 0.5) / sizeof(CacheLineBlock)));

    struct HugeStruct {
        std::uint8_t bytes[1024 * 1024];
    };
    EXPECT_EQ(topo.optimal_l1_block_elements<HugeStruct>(0.5), 0U);

    // Adversarial fractions must never trigger float-cast-overflow UB
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(0.0), 0U);
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(-0.5), 0U);
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(std::numeric_limits<double>::quiet_NaN()), 0U);
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(-std::numeric_limits<double>::infinity()), 0U);
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(2.5),
              static_cast<std::size_t>((l1d * 2.5) / sizeof(int)));
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(1e300), std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(topo.optimal_l1_block_elements<int>(std::numeric_limits<double>::infinity()), 0U);
}

} // namespace
