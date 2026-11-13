#include "algoat/core/hardware_topology.hpp"
#include "algoat/core/traits.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

template <std::size_t Bytes> struct SizedBlock {
    std::uint8_t payload[Bytes];
};

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < sizeof(double)) {
        return 0;
    }

    const auto& topo = algoat::core::HardwareTopology::instance();
    if (&topo != &algoat::core::HardwareTopology::instance()) {
        __builtin_trap();
    }
    if (!std::has_single_bit(topo.cache_line_size()) || topo.l1d_cache_size() == 0U) {
        __builtin_trap();
    }

    double fraction = 0.0;
    std::memcpy(&fraction, data, sizeof(double));

    auto verify_type = [&](auto tag) {
        using T = decltype(tag);
        const std::size_t elems = topo.optimal_l1_block_elements<T>(fraction);
        if (!std::isfinite(fraction) || fraction <= 0.0) {
            if (elems != 0U) {
                __builtin_trap();
            }
        } else {
            const double expected = (static_cast<double>(topo.l1d_cache_size()) * fraction) /
                                    static_cast<double>(sizeof(T));
            constexpr double kMaxSizeTAsDouble =
                static_cast<double>(std::numeric_limits<std::size_t>::max());
            if (!std::isfinite(expected) || expected >= kMaxSizeTAsDouble) {
                if (elems != std::numeric_limits<std::size_t>::max()) {
                    __builtin_trap();
                }
            } else if (elems != static_cast<std::size_t>(expected)) {
                __builtin_trap();
            }
        }
    };

    verify_type(std::uint8_t{});
    verify_type(std::int32_t{});
    verify_type(double{});
    verify_type(SizedBlock<64>{});
    verify_type(SizedBlock<1024>{});
    verify_type(SizedBlock<131072>{});

    // When sufficient fuzzer input is supplied, also exercise sublinear profiler cache-line path
    if (size >= 8U + sizeof(std::int32_t) * 64U) {
        const std::size_t count = (size - 8U) / sizeof(std::int32_t);
        std::vector<std::int32_t> buf(count);
        std::memcpy(buf.data(), data + 8U, count * sizeof(std::int32_t));
        const auto traits = algoat::core::analyze(buf);
        if (traits.size != count || traits.sortedness_ratio < 0.0 ||
            traits.sortedness_ratio > 1.0) {
            __builtin_trap();
        }
    }

    return 0;
}
