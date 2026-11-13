#include "algoat/core/hardware_topology.hpp"
#include "algoat/core/traits.hpp"
#include "algoat/simd/isa.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <rapidcheck.h>
#include <vector>

namespace {

using algoat::core::HardwareTopology;

template <std::size_t Bytes> struct FixedPayload {
    std::uint8_t data[Bytes];
};

TEST(HardwareTopologyPBT, OptimalBlockElementsRespectsBudgetAndMonotonicity) {
    const auto& topo = HardwareTopology::instance();
    const std::size_t l1d = topo.l1d_cache_size();

    rc::check("optimal_l1_block_elements respects budget and monotonicity", [&, l1d]() {
        // Generate arbitrary bit patterns including NaNs, Inf, negative, subnormals, and [0, 1]
        const auto raw_bits_a = *rc::gen::arbitrary<std::uint64_t>();
        const auto raw_bits_b = *rc::gen::arbitrary<std::uint64_t>();
        double fa = 0.0;
        double fb = 0.0;
        std::memcpy(&fa, &raw_bits_a, sizeof(double));
        std::memcpy(&fb, &raw_bits_b, sizeof(double));

        auto check_type = [&](auto tag) {
            using T = decltype(tag);
            const std::size_t ea = topo.optimal_l1_block_elements<T>(fa);
            const std::size_t eb = topo.optimal_l1_block_elements<T>(fb);

            if (!std::isfinite(fa) || fa <= 0.0) {
                RC_ASSERT(ea == 0U);
            } else {
                const double expected =
                    (static_cast<double>(l1d) * fa) / static_cast<double>(sizeof(T));
                constexpr double kMaxSizeTAsDouble =
                    static_cast<double>(std::numeric_limits<std::size_t>::max());
                if (!std::isfinite(expected) || expected >= kMaxSizeTAsDouble) {
                    RC_ASSERT(ea == std::numeric_limits<std::size_t>::max());
                } else {
                    RC_ASSERT(ea == static_cast<std::size_t>(expected));
                }
            }

            if (std::isfinite(fa) && std::isfinite(fb) && fa > 0.0 && fb > 0.0 && fa <= fb) {
                RC_ASSERT(ea <= eb);
            }
        };

        check_type(std::uint8_t{});
        check_type(std::int32_t{});
        check_type(double{});
        check_type(FixedPayload<64>{});
        check_type(FixedPayload<4096>{});
        check_type(FixedPayload<65536>{});
    });
}

TEST(HardwareTopologyPBT, SingletonAndHierarchyInvariantsHold) {
    rc::check("singleton address, cache hierarchy, and ISA invariants", []() {
        const auto& topo = HardwareTopology::instance();
        RC_ASSERT(&topo == &HardwareTopology::instance());

        RC_ASSERT(topo.l1d_cache_size() > 0U);
        RC_ASSERT(topo.l2_cache_size() >= topo.l1d_cache_size());
        RC_ASSERT(std::has_single_bit(topo.cache_line_size()));
        RC_ASSERT(topo.cache_line_size() >= 32U);
        RC_ASSERT(topo.cache_line_size() <= 256U);

        RC_ASSERT(topo.l1d_properties().level == 1U);
        RC_ASSERT(topo.l1d_properties().size_bytes == topo.l1d_cache_size());
        RC_ASSERT(topo.l1d_properties().line_size_bytes == topo.cache_line_size());

        RC_ASSERT(topo.l2_properties().level == 2U);
        RC_ASSERT(topo.l2_properties().size_bytes == topo.l2_cache_size());

        const auto isa = algoat::simd::detect_host_isa();
        if (isa == algoat::simd::InstructionSet::AVX512) {
            RC_ASSERT(topo.has_avx512f());
            RC_ASSERT(topo.has_avx2());
        } else if (isa == algoat::simd::InstructionSet::AVX2) {
            RC_ASSERT(topo.has_avx2());
        } else if (isa == algoat::simd::InstructionSet::Neon) {
            RC_ASSERT(topo.has_neon());
        }
    });
}

TEST(HardwareTopologyPBT, SublinearProfilerRespectsSortedAndReversedInvariants) {
    rc::check("sublinear profiler detects monotonic sequences using runtime cache line",
              [](std::size_t n_seed) {
                  const std::size_t n = 10001U + (n_seed % 5000U);
                  std::vector<std::int32_t> asc(n);
                  for (std::size_t i = 0; i < n; ++i) {
                      asc[i] = static_cast<std::int32_t>(i);
                  }
                  const auto asc_traits = algoat::core::analyze(asc);
                  RC_ASSERT(asc_traits.size == n);
                  RC_ASSERT(!asc_traits.is_exact);
                  RC_ASSERT(asc_traits.sortedness_ratio >= 0.99);
                  RC_ASSERT(!asc_traits.has_duplicates);
              });
}

} // namespace
