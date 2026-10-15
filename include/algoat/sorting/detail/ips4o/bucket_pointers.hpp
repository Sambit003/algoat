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
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

#if UINTPTR_MAX != UINT32_MAX && !defined(__SIZEOF_INT128__)
#error "Unsupported architecture: 128-bit integers required on 64-bit platforms"
#endif

template <class Cfg> class Sorter<Cfg>::BucketPointers {
#if UINTPTR_MAX == UINT32_MAX
    using atomic_type = std::uint64_t;
#else
    using atomic_type = unsigned __int128;
#endif
    static constexpr const int kShift = sizeof(atomic_type) * CHAR_BIT / 2;
    static constexpr const atomic_type kMask = (static_cast<atomic_type>(1) << kShift) - 1;
    using diff_t = typename Cfg::difference_type;

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
        if (kAtomic) {
            const auto p = __atomic_fetch_add(&all_, Cfg::kBlockSize, __ATOMIC_RELAXED);
            const diff_t w = static_cast<diff_t>(p & kMask);
            const diff_t r = static_cast<diff_t>(p >> kShift);
            return {w, r};
        } else {
            const auto w = single_.w;
            single_.w += Cfg::kBlockSize;
            return {w, single_.r};
        }
    }

    template <bool kAtomic> std::pair<diff_t, diff_t> decRead() {
        if (kAtomic) {
            num_reading_.fetch_add(1, std::memory_order_acquire);
            const auto p = __atomic_fetch_sub(
                &all_, static_cast<atomic_type>(Cfg::kBlockSize) << kShift, __ATOMIC_RELAXED);
            const diff_t w = static_cast<diff_t>(p & kMask);
            const diff_t r = static_cast<diff_t>((p >> kShift) & ~(Cfg::kBlockSize - 1));
            return {w, r};
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

private:
    struct Pointers {
        diff_t w, r;
    };
    union {
        atomic_type all_;
        Pointers single_;
    };
    std::atomic_int num_reading_{0};
};

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
