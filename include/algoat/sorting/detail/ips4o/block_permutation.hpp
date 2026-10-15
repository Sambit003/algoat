/******************************************************************************
 * algoat/sorting/detail/ips4o/block_permutation.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "classifier.hpp"
#include "ips4o_fwd.hpp"
#include "memory.hpp"

#include <tuple>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class Cfg> int Sorter<Cfg>::computeOverflowBucket() {
    int bucket = num_buckets_ - 1;
    while (bucket >= 0 && (bucket_start_[bucket + 1] - bucket_start_[bucket]) <= Cfg::kBlockSize)
        --bucket;
    return bucket;
}

template <class Cfg>
template <bool kEqualBuckets, bool kIsParallel>
int Sorter<Cfg>::classifyAndReadBlock(const int read_bucket) {
    auto& bp = bucket_pointers_[read_bucket];

    diff_t write, read;
    std::tie(write, read) = bp.template decRead<kIsParallel>();

    if (read < write) {
        if (kIsParallel)
            bp.stopRead();
        return -1;
    }

    local_.swap[0].readFrom(begin_ + read);
    if (kIsParallel)
        bp.stopRead();

    return classifier_->template classify<kEqualBuckets>(local_.swap[0].head());
}

template <class Cfg>
template <bool kEqualBuckets, bool kIsParallel>
int Sorter<Cfg>::swapBlock(const diff_t max_off, const int dest_bucket, const bool current_swap) {
    diff_t write, read;
    int new_dest_bucket;
    auto& bp = bucket_pointers_[dest_bucket];
    do {
        std::tie(write, read) = bp.template incWrite<kIsParallel>();
        if (write > read) {
            if (write >= max_off) {
                local_.swap[current_swap].writeTo(local_.overflow);
                overflow_ = &local_.overflow;
                return -1;
            }
            while (kIsParallel && bp.isReading()) {
            }
            local_.swap[current_swap].writeTo(begin_ + write);
            return -1;
        }
        new_dest_bucket = classifier_->template classify<kEqualBuckets>(begin_[write]);
    } while (new_dest_bucket == dest_bucket);

    local_.swap[!current_swap].readFrom(begin_ + write);
    local_.swap[current_swap].writeTo(begin_ + write);

    return new_dest_bucket;
}

template <class Cfg>
template <bool kEqualBuckets, bool kIsParallel>
void Sorter<Cfg>::permuteBlocks() {
    const auto num_buckets = num_buckets_;
    int read_bucket = (my_id_ * num_buckets / num_threads_) % num_buckets;
    const diff_t max_off = Cfg::alignToNextBlock(end_ - begin_ + 1) - Cfg::kBlockSize;

    for (int count = num_buckets; count; --count) {
        int dest_bucket;
        while ((dest_bucket = classifyAndReadBlock<kEqualBuckets, kIsParallel>(read_bucket)) !=
               -1) {
            bool current_swap = false;
            while ((dest_bucket = swapBlock<kEqualBuckets, kIsParallel>(max_off, dest_bucket,
                                                                        current_swap)) != -1) {
                current_swap = !current_swap;
            }
        }
        read_bucket = (read_bucket + 1) % num_buckets;
    }
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
