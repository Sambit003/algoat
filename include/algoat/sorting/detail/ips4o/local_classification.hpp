/******************************************************************************
 * algoat/sorting/detail/ips4o/local_classification.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "classifier.hpp"
#include "empty_block_movement.hpp"
#include "ips4o_fwd.hpp"
#include "memory.hpp"

#include <atomic>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class Cfg>
template <bool kEqualBuckets>
typename Cfg::difference_type Sorter<Cfg>::classifyLocally(const iterator my_begin,
                                                           const iterator my_end) {
    auto write = my_begin;
    auto& buffers = local_.buffers;

    classifier_->template classify<kEqualBuckets>(
        my_begin, my_end, [&](typename Cfg::bucket_type bucket, iterator it) {
            if (buffers.isFull(static_cast<int>(bucket))) {
                buffers.writeTo(static_cast<int>(bucket), write);
                write += Cfg::kBlockSize;
                local_.bucket_size[bucket] += Cfg::kBlockSize;
            }
            buffers.push(static_cast<int>(bucket), std::move(*it));
        });

    for (int i = 0, end = num_buckets_; i < end; ++i)
        local_.bucket_size[i] += local_.buffers.size(i);

    return write - begin_;
}

template <class Cfg> void Sorter<Cfg>::sequentialClassification(const bool use_equal_buckets) {
    const auto my_first_empty_block = use_equal_buckets ? classifyLocally<true>(begin_, end_)
                                                        : classifyLocally<false>(begin_, end_);

    diff_t sum = 0;
    bucket_start_[0] = 0;
    for (int i = 0, end = num_buckets_; i < end; ++i) {
        sum += local_.bucket_size[i];
        bucket_start_[i + 1] = sum;
    }
    IPS4O_ASSUME_NOT(bucket_start_[num_buckets_] != end_ - begin_);

    for (int bucket = 0, end = num_buckets_; bucket < end; ++bucket) {
        const auto start = Cfg::alignToNextBlock(bucket_start_[bucket]);
        const auto stop = Cfg::alignToNextBlock(bucket_start_[bucket + 1]);
        bucket_pointers_[bucket].set(
            start, (start >= my_first_empty_block
                        ? start
                        : (stop <= my_first_empty_block ? stop : my_first_empty_block)) -
                       Cfg::kBlockSize);
    }
}

template <class Cfg> void Sorter<Cfg>::parallelClassification(const bool use_equal_buckets) {
    const auto elements_per_thread = static_cast<double>(end_ - begin_) / num_threads_;
    const auto my_begin =
        begin_ + Cfg::alignToNextBlock(static_cast<diff_t>(my_id_ * elements_per_thread + 0.5));
    const auto my_end = [&] {
        const auto size = end_ - begin_;
        const auto e =
            Cfg::alignToNextBlock(static_cast<diff_t>((my_id_ + 1) * elements_per_thread + 0.5));
        if (size < e) {
            return end_;
        }
        return begin_ + e;
    }();

    local_.first_block = my_begin - begin_;

    if (my_begin >= my_end) {
        local_.first_empty_block = my_begin - begin_;
        shared_->sync.barrier();
    } else {
        const auto my_first_empty_block = use_equal_buckets
                                              ? classifyLocally<true>(my_begin, my_end)
                                              : classifyLocally<false>(my_begin, my_end);

        diff_t sum = 0;
        for (int i = 0, end = num_buckets_; i < end; ++i) {
            sum += local_.bucket_size[i];
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
            std::atomic_ref<diff_t>(bucket_start_[i + 1]).fetch_add(sum, std::memory_order_relaxed);
#elif defined(_MSC_VER)
            if constexpr (sizeof(diff_t) == 8) {
                _InterlockedAdd64(reinterpret_cast<volatile __int64*>(&bucket_start_[i + 1]), sum);
            } else {
                _InterlockedAdd(reinterpret_cast<volatile long*>(&bucket_start_[i + 1]), sum);
            }
#else
            __atomic_fetch_add(&bucket_start_[i + 1], sum, __ATOMIC_RELAXED);
#endif
        }

        local_.first_empty_block = my_first_empty_block;

        shared_->sync.barrier();

        moveEmptyBlocks(my_begin - begin_, my_end - begin_, my_first_empty_block);
    }

    shared_->sync.barrier();
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
