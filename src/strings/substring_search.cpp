#include "algoat/strings/substring_search.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "strings/substring_search.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace algoat::strings::detail {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

std::string_view::size_type find_substring_simd_impl(const char* hs_data, std::size_t n,
                                                     const char* ne_data, std::size_t m) {
    if (m == 0) {
        return 0;
    }
    if (n < m) {
        return std::string_view::npos;
    }

    const hn::ScalableTag<uint8_t> d;
    const std::size_t N = hn::Lanes(d);

    if (n < N + m - 1) {
        return detail::two_way_search(std::string_view(hs_data, n), std::string_view(ne_data, m));
    }

    const auto v_first = hn::Set(d, static_cast<uint8_t>(ne_data[0]));
    const auto v_last = hn::Set(d, static_cast<uint8_t>(ne_data[m - 1]));

    const auto* const hs_ptr = reinterpret_cast<const uint8_t*>(hs_data);
    const auto* const ne_ptr = reinterpret_cast<const uint8_t*>(ne_data);

    std::size_t i = 0;
    const std::size_t simd_limit = n - (N + m - 1);

    std::size_t pathological_streak = 0;
    constexpr std::size_t kMaxPathologicalStreak = 8;

    while (i <= simd_limit) {
        const auto v0 = hn::LoadU(d, hs_ptr + i);
        const auto v1 = hn::LoadU(d, hs_ptr + i + m - 1);
        const auto mask = hn::And(hn::Eq(v0, v_first), hn::Eq(v1, v_last));

        if (hn::AllFalse(d, mask)) {
            pathological_streak = 0;
            i += N;
            continue;
        }

        if (hn::AllTrue(d, mask)) {
            ++pathological_streak;
            if (pathological_streak > kMaxPathologicalStreak) {
                // Dense collision attack detected: switch remaining search to Two-Way for O(N)
                // guarantee
                const auto tail_res = detail::two_way_search(std::string_view(hs_data + i, n - i),
                                                             std::string_view(ne_data, m));
                return (tail_res != std::string_view::npos) ? (i + tail_res)
                                                            : std::string_view::npos;
            }
        } else {
            pathological_streak = 0;
        }

        alignas(8) uint8_t bit_buf[64] = {0};
        const std::size_t num_bytes = hn::StoreMaskBits(d, mask, bit_buf);
        for (std::size_t byte_idx = 0; byte_idx < num_bytes; byte_idx += 8) {
            uint64_t mask_bits = 0;
            const std::size_t bytes_to_copy = std::min<std::size_t>(8, num_bytes - byte_idx);
            std::memcpy(&mask_bits, bit_buf + byte_idx, bytes_to_copy);
            while (mask_bits != 0) {
                const uint32_t lane = std::countr_zero(mask_bits);
                const std::size_t candidate_idx = i + byte_idx * 8 + lane;
                if (m <= 2 || std::memcmp(hs_ptr + candidate_idx + 1, ne_ptr + 1, m - 2) == 0) {
                    return candidate_idx;
                }
                mask_bits &= (mask_bits - 1);
            }
        }
        i += N;
    }

    // Process remaining tail positions using scalar Two-Way search
    if (i <= n - m) {
        const auto tail_res = detail::two_way_search(std::string_view(hs_data + i, n - i),
                                                     std::string_view(ne_data, m));
        if (tail_res != std::string_view::npos) {
            return i + tail_res;
        }
    }

    return std::string_view::npos;
}

} // namespace HWY_NAMESPACE
} // namespace algoat::strings::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace algoat::strings::detail {
HWY_EXPORT(find_substring_simd_impl);

std::string_view::size_type find_substring_simd_dispatch(const char* hs_data, std::size_t n,
                                                         const char* ne_data,
                                                         std::size_t m) noexcept {
    if (m == 0) {
        return 0;
    }
    if (n < m) {
        return std::string_view::npos;
    }
    return HWY_DYNAMIC_DISPATCH(find_substring_simd_impl)(hs_data, n, ne_data, m);
}

} // namespace algoat::strings::detail
#endif
