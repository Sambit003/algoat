/******************************************************************************
 * algoat/sorting/detail/ips4o/cleanup_margins.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "base_case.hpp"
#include "ips4o_fwd.hpp"
#include "memory.hpp"
#include "utils.hpp"

#include <limits>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class Cfg>
std::pair<int, typename Cfg::difference_type> Sorter<Cfg>::saveMargins(int last_bucket) {
    diff_t tail = bucket_start_[last_bucket];
    const diff_t end = Cfg::alignToNextBlock(tail);

    if (tail == end || end > (end_ - begin_))
        return {-1, 0};

    {
        const auto start_of_last_block = end - Cfg::kBlockSize;
        diff_t last_start;
        do {
            --last_bucket;
            last_start = bucket_start_[last_bucket];
        } while (last_start > start_of_last_block);
    }

    const auto write = shared_->bucket_pointers[last_bucket].getWrite();
    if (write < end)
        return {-1, 0};

    tail = bucket_start_[last_bucket + 1];
    local_.swap[0].readFrom(begin_ + tail, end - tail);

    return {last_bucket, end - tail};
}

template <class Cfg>
void Sorter<Cfg>::writeMargins(const int first_bucket, const int last_bucket,
                               const int overflow_bucket, const int swap_bucket,
                               const diff_t in_swap_buffer) {
    const bool is_last_level = end_ - begin_ <= Cfg::kSingleLevelThreshold;
    const auto comp = classifier_->getComparator();

    for (int i = first_bucket; i < last_bucket; ++i) {
        const auto bstart = bucket_start_[i];
        const auto bend = bucket_start_[i + 1];
        const auto bwrite = bucket_pointers_[i].getWrite();
        auto dst = begin_ + bstart;
        auto remaining = Cfg::alignToNextBlock(bstart) - bstart;

        if (i == overflow_bucket && overflow_) {
            IPS4O_ASSUME_NOT(Cfg::alignToNextBlock(bend) != bwrite);

            auto src = overflow_->data();
            IPS4O_ASSUME_NOT((bend - (bwrite - Cfg::kBlockSize)) + remaining < Cfg::kBlockSize);
            auto tail_size = Cfg::kBlockSize - remaining;

            std::move(src, src + remaining, dst);
            src += remaining;
            remaining = std::numeric_limits<diff_t>::max();

            dst = begin_ + (bwrite - Cfg::kBlockSize);
            dst = std::move(src, src + tail_size, dst);

            overflow_->reset(Cfg::kBlockSize);
        } else if (i == swap_bucket && in_swap_buffer) {
            auto src = local_.swap[0].data();
            IPS4O_ASSUME_NOT(in_swap_buffer > remaining);

            dst = std::move(src, src + in_swap_buffer, dst);
            remaining -= in_swap_buffer;

            local_.swap[0].reset(in_swap_buffer);
        } else if (bwrite > bend && bend - bstart > Cfg::kBlockSize) {
            IPS4O_ASSUME_NOT(Cfg::alignToNextBlock(bend) != bwrite);

            auto src = begin_ + bend;
            auto head_size = bwrite - bend;
            IPS4O_ASSUME_NOT(head_size > remaining);

            dst = std::move(src, src + head_size, dst);
            remaining -= head_size;
        }

        for (int t = 0; t < num_threads_; ++t) {
            auto& buffers = shared_ ? shared_->local[t]->buffers : local_.buffers;
            auto src = buffers.data(i);
            auto count = buffers.size(i);

            if (count <= remaining) {
                dst = std::move(src, src + count, dst);
                remaining -= count;
            } else {
                std::move(src, src + remaining, dst);
                src += remaining;
                count -= remaining;
                remaining = std::numeric_limits<diff_t>::max();

                dst = begin_ + bwrite;
                dst = std::move(src, src + count, dst);
            }

            buffers.reset(i);
        }

        if (is_last_level || (bend - bstart) <= 2 * Cfg::kBaseCaseSize)
            detail::baseCaseSort(begin_ + bstart, begin_ + bend, comp);
    }
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
