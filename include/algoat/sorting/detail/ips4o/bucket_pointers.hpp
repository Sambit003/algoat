/******************************************************************************
 * algoat/sorting/detail/ips4o/bucket_pointers.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "ips4o_fwd.hpp"

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64) || defined(_M_ARM64)
#pragma intrinsic(_InterlockedCompareExchange128)
#elif defined(_M_IX86)
#pragma intrinsic(_InterlockedCompareExchange64)
#endif
#endif

namespace algoat::sorting::detail::ips4o {
namespace detail {

#if UINTPTR_MAX != UINT32_MAX && !defined(__SIZEOF_INT128__) && !defined(_MSC_VER)
#error "Unsupported architecture: 128-bit atomic operations required on 64-bit platforms"
#endif

template <class Cfg> class Sorter<Cfg>::BucketPointers {
    using diff_t = typename Cfg::difference_type;

    struct alignas(2 * sizeof(diff_t)) Pointers {
        diff_t w;
        diff_t r;
    };

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    static_assert(sizeof(diff_t) == sizeof(__int64),
                  "difference_type must be 64-bit on 64-bit MSVC");
    alignas(16) Pointers single_{0, 0};
#elif defined(_MSC_VER) && defined(_M_IX86)
    static_assert(sizeof(diff_t) == sizeof(std::int32_t),
                  "difference_type must be 32-bit on 32-bit MSVC");
    alignas(8) Pointers single_{0, 0};
#elif UINTPTR_MAX == UINT32_MAX
    using atomic_type = std::uint64_t;
    static constexpr const int kShift = sizeof(atomic_type) * CHAR_BIT / 2;
    static constexpr const atomic_type kMask = (static_cast<atomic_type>(1) << kShift) - 1;
    union {
        atomic_type all_{0};
        Pointers single_;
    };
#else
    using atomic_type = unsigned __int128;
    static constexpr const int kShift = sizeof(atomic_type) * CHAR_BIT / 2;
    static constexpr const atomic_type kMask = (static_cast<atomic_type>(1) << kShift) - 1;
    union {
        atomic_type all_{0};
        Pointers single_;
    };
#endif

    std::atomic_int num_reading_{0};

public:
    void set(diff_t w, diff_t r) {
        single_.w = w;
        single_.r = r;
        num_reading_.store(0, std::memory_order_relaxed);
    }

    diff_t getWrite() const {
        return single_.w;
    }

    template <bool kAtomic> std::pair<diff_t, diff_t> incWrite() {
        if constexpr (kAtomic) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
            alignas(16) __int64 comparand[2];
            comparand[0] = static_cast<__int64>(single_.w);
            comparand[1] = static_cast<__int64>(single_.r);
            while (!_InterlockedCompareExchange128(
                reinterpret_cast<volatile __int64*>(&single_),
                comparand[1], /* ExchangeHigh: keep r */
                comparand[0] +
                    static_cast<__int64>(Cfg::kBlockSize), /* ExchangeLow: w + kBlockSize */
                comparand)) {
            }
            return {static_cast<diff_t>(comparand[0]), static_cast<diff_t>(comparand[1])};
#elif defined(_MSC_VER) && defined(_M_IX86)
            __int64 expected;
            std::memcpy(&expected, &single_, sizeof(__int64));
            while (true) {
                const diff_t cur_w = static_cast<diff_t>(expected & 0xFFFFFFFFULL);
                const diff_t cur_r = static_cast<diff_t>((expected >> 32) & 0xFFFFFFFFULL);
                const __int64 desired = (static_cast<__int64>(cur_r) << 32) |
                                        static_cast<std::uint32_t>(cur_w + Cfg::kBlockSize);
                const __int64 old = _InterlockedCompareExchange64(
                    reinterpret_cast<volatile __int64*>(&single_), desired, expected);
                if (old == expected) {
                    return {cur_w, cur_r};
                }
                expected = old;
            }
#else
            const auto p = __atomic_fetch_add(&all_, Cfg::kBlockSize, __ATOMIC_RELAXED);
            const diff_t w = static_cast<diff_t>(p & kMask);
            const diff_t r = static_cast<diff_t>(p >> kShift);
            return {w, r};
#endif
        } else {
            const auto w = single_.w;
            single_.w += Cfg::kBlockSize;
            return {w, single_.r};
        }
    }

    template <bool kAtomic> std::pair<diff_t, diff_t> decRead() {
        if constexpr (kAtomic) {
            num_reading_.fetch_add(1, std::memory_order_acquire);
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
            alignas(16) __int64 comparand[2];
            comparand[0] = static_cast<__int64>(single_.w);
            comparand[1] = static_cast<__int64>(single_.r);
            while (!_InterlockedCompareExchange128(
                reinterpret_cast<volatile __int64*>(&single_),
                comparand[1] -
                    static_cast<__int64>(Cfg::kBlockSize), /* ExchangeHigh: r - kBlockSize */
                comparand[0],                              /* ExchangeLow: keep w */
                comparand)) {
            }
            const diff_t w = static_cast<diff_t>(comparand[0]);
            const diff_t r = static_cast<diff_t>(comparand[1] & ~(Cfg::kBlockSize - 1));
            return {w, r};
#elif defined(_MSC_VER) && defined(_M_IX86)
            __int64 expected;
            std::memcpy(&expected, &single_, sizeof(__int64));
            while (true) {
                const diff_t cur_w = static_cast<diff_t>(expected & 0xFFFFFFFFULL);
                const diff_t cur_r = static_cast<diff_t>((expected >> 32) & 0xFFFFFFFFULL);
                const __int64 desired = (static_cast<__int64>(cur_r - Cfg::kBlockSize) << 32) |
                                        static_cast<std::uint32_t>(cur_w);
                const __int64 old = _InterlockedCompareExchange64(
                    reinterpret_cast<volatile __int64*>(&single_), desired, expected);
                if (old == expected) {
                    const diff_t r = static_cast<diff_t>(cur_r & ~(Cfg::kBlockSize - 1));
                    return {cur_w, r};
                }
                expected = old;
            }
#else
            const auto p = __atomic_fetch_sub(
                &all_, static_cast<atomic_type>(Cfg::kBlockSize) << kShift, __ATOMIC_RELAXED);
            const diff_t w = static_cast<diff_t>(p & kMask);
            const diff_t r = static_cast<diff_t>((p >> kShift) & ~(Cfg::kBlockSize - 1));
            return {w, r};
#endif
        } else {
            const auto r = single_.r;
            single_.r -= Cfg::kBlockSize;
            return {single_.w, r};
        }
    }

    void stopRead() {
        num_reading_.fetch_sub(1, std::memory_order_release);
    }

    bool isReading() {
        return num_reading_.load(std::memory_order_acquire) != 0;
    }
};

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
