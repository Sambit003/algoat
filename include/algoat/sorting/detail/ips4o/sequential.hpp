/******************************************************************************
 * algoat/sorting/detail/ips4o/sequential.hpp
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
#include "partitioning.hpp"

#include <tuple>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class Cfg> void Sorter<Cfg>::sequential(const iterator begin, const iterator end) {
    const auto n = end - begin;
    if (n <= 2 * Cfg::kBaseCaseSize) {
        detail::baseCaseSort(begin, end, local_.classifier.getComparator());
        return;
    }
    diff_t bucket_start[Cfg::kMaxBuckets + 1];

    const auto res = partition<false>(begin, end, bucket_start, nullptr, 0, 1);
    const int num_buckets = std::get<0>(res);
    const bool equal_buckets = std::get<1>(res);

    if (n <= Cfg::kSingleLevelThreshold) {
        return;
    }

    for (int i = 0; i < num_buckets; i += 1 + equal_buckets) {
        const auto start = bucket_start[i];
        const auto stop = bucket_start[i + 1];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
            sequential(begin + start, begin + stop);
    }
    if (equal_buckets) {
        const auto start = bucket_start[num_buckets - 1];
        const auto stop = bucket_start[num_buckets];
        if (stop - start > 2 * Cfg::kBaseCaseSize)
            sequential(begin + start, begin + stop);
    }
}

} // namespace detail

template <class Cfg> class SequentialSorter {
    using Sorter = detail::Sorter<Cfg>;
    using iterator = typename Cfg::iterator;

public:
    explicit SequentialSorter(bool check_sorted, typename Cfg::less comp)
        : check_sorted_(check_sorted), buffer_storage_(1),
          local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage_.get()) {}

    explicit SequentialSorter(bool check_sorted, typename Cfg::less comp, char* buffer_storage)
        : check_sorted_(check_sorted),
          local_ptr_(Cfg::kDataAlignment, std::move(comp), buffer_storage) {}

    void operator()(iterator begin, iterator end) {
        if (check_sorted_) {
            const bool sorted =
                detail::sortedCaseSort(begin, end, local_ptr_.get().classifier.getComparator());
            if (sorted)
                return;
        }

        Sorter(local_ptr_.get()).sequential(std::move(begin), std::move(end));
    }

private:
    const bool check_sorted_;
    typename Sorter::BufferStorage buffer_storage_;
    detail::AlignedPtr<typename Sorter::LocalData> local_ptr_;
};

} // namespace algoat::sorting::detail::ips4o
