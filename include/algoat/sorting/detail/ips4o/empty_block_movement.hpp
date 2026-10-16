/******************************************************************************
 * algoat/sorting/detail/ips4o/empty_block_movement.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "ips4o_fwd.hpp"
#include "memory.hpp"

#include <algorithm>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class Cfg>
void Sorter<Cfg>::moveEmptyBlocks(const diff_t my_begin, const diff_t my_end,
                                  const diff_t my_first_empty_block) {
    const int bucket_range_start = [&](int i) {
        while (Cfg::alignToNextBlock(bucket_start_[i]) < my_begin)
            ++i;
        return i;
    }(0);
    const int bucket_range_end = [&](int i) {
        const auto num_buckets = num_buckets_;
        if (my_id_ == num_threads_ - 1)
            return num_buckets;
        while (i < num_buckets && Cfg::alignToNextBlock(bucket_start_[i]) < my_end)
            ++i;
        return i;
    }(bucket_range_start);

    const auto bucket_end = Cfg::alignToNextBlock(bucket_start_[bucket_range_end]);
    const bool last_bucket_is_overlapping = bucket_end > my_end;

    for (int b = bucket_range_start; b < bucket_range_end - last_bucket_is_overlapping; ++b) {
        const auto start = Cfg::alignToNextBlock(bucket_start_[b]);
        const auto stop = Cfg::alignToNextBlock(bucket_start_[b + 1]);
        auto read = stop;
        if (my_first_empty_block <= start) {
            read = start;
        } else if (my_first_empty_block < stop) {
            read = my_first_empty_block;
        }
        bucket_pointers_[b].set(start, read - Cfg::kBlockSize);
    }

    if (last_bucket_is_overlapping) {
        const int overlapping_bucket = bucket_range_end - 1;
        const auto bucket_start = Cfg::alignToNextBlock(bucket_start_[overlapping_bucket]);

        diff_t flushed_elements_in_bucket = 0;
        if (bucket_start < my_begin) {
            int prev_id = my_id_ - 1;
            while (bucket_start < shared_->local[prev_id]->first_block) {
                const auto eb = shared_->local[prev_id]->first_empty_block;
                flushed_elements_in_bucket += eb - shared_->local[prev_id]->first_block;
                --prev_id;
            }
            const auto eb = shared_->local[prev_id]->first_empty_block;
            if (eb > bucket_start)
                flushed_elements_in_bucket += eb - bucket_start;
        }

        diff_t elements_reserved = 0;
        if (my_begin > bucket_start) {
            elements_reserved = my_begin - bucket_start - flushed_elements_in_bucket;
            flushed_elements_in_bucket += my_first_empty_block - my_begin;
        } else if (my_first_empty_block > bucket_start) {
            flushed_elements_in_bucket += my_first_empty_block - bucket_start;
        }

        int read_from_thread = my_id_ + 1;
        while (read_from_thread < num_threads_ &&
               bucket_end > shared_->local[read_from_thread]->first_block) {
            const auto eb =
                std::min<diff_t>(shared_->local[read_from_thread]->first_empty_block, bucket_end);
            flushed_elements_in_bucket += eb - shared_->local[read_from_thread]->first_block;
            ++read_from_thread;
        }

        const auto first_empty_block_in_bucket = bucket_start + flushed_elements_in_bucket;

        auto write_ptr = begin_ + std::max(my_first_empty_block, bucket_start);
        const auto write_ptr_end = begin_ + std::min(first_empty_block_in_bucket, my_end);

        while (write_ptr < write_ptr_end) {
            --read_from_thread;
            auto read_ptr =
                std::min(shared_->local[read_from_thread]->first_empty_block, bucket_end);
            auto read_range_size = read_ptr - shared_->local[read_from_thread]->first_block;

            if (elements_reserved >= read_range_size) {
                elements_reserved -= read_range_size;
                continue;
            }
            read_ptr -= elements_reserved;
            read_range_size -= elements_reserved;
            elements_reserved = 0;

            const auto size =
                std::min(read_range_size, static_cast<diff_t>(write_ptr_end - write_ptr));
            write_ptr = std::move(begin_ + read_ptr - size, begin_ + read_ptr, write_ptr);
        }

        if (my_begin <= bucket_start) {
            bucket_pointers_[overlapping_bucket].set(bucket_start,
                                                     first_empty_block_in_bucket - Cfg::kBlockSize);
        }
    }
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
