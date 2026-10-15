/******************************************************************************
 * algoat/sorting/detail/ips4o/sampling.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "algoat/sorting/pdqsort.hpp"
#include "classifier.hpp"
#include "config.hpp"
#include "ips4o_fwd.hpp"
#include "memory.hpp"

#include <iterator>
#include <new>
#include <random>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class It, class RandomGen>
void selectSample(It begin, const It end,
                  typename std::iterator_traits<It>::difference_type num_samples, RandomGen&& gen) {
    using std::swap;

    auto n = end - begin;
    while (num_samples--) {
        const auto i =
            std::uniform_int_distribution<typename std::iterator_traits<It>::difference_type>(
                0, --n)(gen);
        swap(*begin, begin[i]);
        ++begin;
    }
}

template <class Cfg>
std::pair<int, bool> Sorter<Cfg>::buildClassifier(const iterator begin, const iterator end,
                                                  Classifier& classifier) {
    const auto n = end - begin;
    int log_buckets = Cfg::logBuckets(n);
    int num_buckets = 1 << log_buckets;
    const auto step = std::max<diff_t>(1, static_cast<diff_t>(Cfg::oversamplingFactor(n)));
    const auto num_samples = step * num_buckets - 1;

    detail::selectSample(begin, end, num_samples, local_.random_generator);

    ::algoat::sorting::pdqsort(begin, begin + num_samples, classifier.getComparator());
    auto splitter = begin + step - 1;
    auto sorted_splitters = classifier.getSortedSplitters();
    const auto comp = classifier.getComparator();

    IPS4O_ASSUME_NOT(sorted_splitters == nullptr);
    new (static_cast<void*>(sorted_splitters)) typename Cfg::value_type(*splitter);
    for (int i = 2; i < num_buckets; ++i) {
        splitter += step;
        if (comp(*sorted_splitters, *splitter)) {
            IPS4O_ASSUME_NOT(sorted_splitters + 1 == nullptr);
            new (static_cast<void*>(++sorted_splitters)) typename Cfg::value_type(*splitter);
        }
    }

    const auto diff_splitters = sorted_splitters - classifier.getSortedSplitters() + 1;
    const bool use_equal_buckets =
        Cfg::kAllowEqualBuckets && num_buckets - 1 - diff_splitters >= Cfg::kEqualBucketsThreshold;

    log_buckets = static_cast<int>(log2(static_cast<unsigned long>(diff_splitters)) + 1);
    num_buckets = 1 << log_buckets;
    for (int i = diff_splitters + 1; i < num_buckets; ++i) {
        IPS4O_ASSUME_NOT(sorted_splitters + 1 == nullptr);
        new (static_cast<void*>(++sorted_splitters)) typename Cfg::value_type(*splitter);
    }

    classifier.build(log_buckets);
    this->classifier_ = &classifier;

    const int used_buckets = num_buckets * (1 + use_equal_buckets);
    return {used_buckets, use_equal_buckets};
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
