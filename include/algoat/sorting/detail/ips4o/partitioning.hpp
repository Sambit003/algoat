/******************************************************************************
 * algoat/sorting/detail/ips4o/partitioning.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "block_permutation.hpp"
#include "cleanup_margins.hpp"
#include "ips4o_fwd.hpp"
#include "local_classification.hpp"
#include "memory.hpp"
#include "sampling.hpp"
#include "utils.hpp"

#include <atomic>
#include <tuple>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class Cfg>
template <bool kIsParallel>
std::pair<int, bool> Sorter<Cfg>::partition(const iterator begin, const iterator end,
                                            diff_t* const bucket_start, SharedData* const shared,
                                            const int my_id, const int num_threads) {
    bool use_equal_buckets = false;
    {
        if constexpr (!kIsParallel) {
            std::tie(this->num_buckets_, use_equal_buckets) =
                buildClassifier(begin, end, local_.classifier);
        } else {
            shared->sync.single([&] {
                std::tie(this->num_buckets_, use_equal_buckets) =
                    buildClassifier(begin, end, shared->classifier);
                shared->num_buckets = this->num_buckets_;
                shared->use_equal_buckets = use_equal_buckets;
            });
            this->num_buckets_ = shared->num_buckets;
            use_equal_buckets = shared->use_equal_buckets;
        }
    }

    this->shared_ = shared;
    this->classifier_ = kIsParallel ? &shared_->classifier : &local_.classifier;
    this->bucket_start_ = bucket_start;
    this->bucket_pointers_ = kIsParallel ? shared_->bucket_pointers : local_.bucket_pointers;
    this->overflow_ = nullptr;
    this->begin_ = begin;
    this->end_ = end;
    this->my_id_ = my_id;
    this->num_threads_ = num_threads;

    if constexpr (kIsParallel)
        parallelClassification(use_equal_buckets);
    else
        sequentialClassification(use_equal_buckets);

    const int overflow_bucket = computeOverflowBucket();

    if (use_equal_buckets)
        permuteBlocks<true, kIsParallel>();
    else
        permuteBlocks<false, kIsParallel>();

    if constexpr (kIsParallel) {
        if (overflow_)
            shared_->overflow = &local_.overflow;
        shared_->sync.barrier();
    }

    {
        if constexpr (kIsParallel)
            overflow_ = shared_->overflow;

        const int num_buckets = num_buckets_;
        const int buckets_per_thread = (num_buckets + num_threads_ - 1) / num_threads_;
        int my_first_bucket = my_id_ * buckets_per_thread;
        int my_last_bucket = (my_id_ + 1) * buckets_per_thread;
        my_first_bucket = num_buckets < my_first_bucket ? num_buckets : my_first_bucket;
        my_last_bucket = num_buckets < my_last_bucket ? num_buckets : my_last_bucket;

        const auto in_swap_buffer =
            !kIsParallel ? std::pair<int, diff_t>(-1, 0) : saveMargins(my_last_bucket);
        if constexpr (kIsParallel)
            shared_->sync.barrier();

        writeMargins(my_first_bucket, my_last_bucket, overflow_bucket, in_swap_buffer.first,
                     in_swap_buffer.second);
    }

    if constexpr (kIsParallel)
        shared_->sync.barrier();
    local_.reset();

    return {this->num_buckets_, use_equal_buckets};
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
