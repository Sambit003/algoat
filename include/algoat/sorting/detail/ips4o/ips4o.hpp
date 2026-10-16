/******************************************************************************
 * algoat/sorting/detail/ips4o/ips4o.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "base_case.hpp"
#include "config.hpp"
#include "ips4o_fwd.hpp"
#include "memory.hpp"
#include "parallel.hpp"
#include "sequential.hpp"

#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

namespace algoat::sorting::detail::ips4o {

template <class It, class Cfg = Config<>, class Comp = std::less<>>
SequentialSorter<ExtendedConfig<It, Comp, Cfg>> make_sorter(Comp comp = Comp()) {
    return SequentialSorter<ExtendedConfig<It, Comp, Cfg>>{true, std::move(comp)};
}

template <class Cfg = Config<>, class It, class Comp = std::less<>>
void sort(It begin, It end, Comp comp = Comp()) {
    if (detail::sortedCaseSort(begin, end, comp))
        return;

    if ((end - begin) <= Cfg::kBaseCaseMultiplier * Cfg::kBaseCaseSize) {
        detail::baseCaseSort(std::move(begin), std::move(end), std::move(comp));
    } else {
        SequentialSorter<ExtendedConfig<It, Comp, Cfg>> sorter{false, std::move(comp)};
        sorter(std::move(begin), std::move(end));
    }
}

namespace parallel {

template <class It, class Cfg = Config<>, class ThreadPool, class Comp = std::less<>>
ParallelSorter<ExtendedConfig<It, Comp, Cfg, std::decay_t<ThreadPool>>>
make_sorter(ThreadPool&& thread_pool, Comp comp = Comp()) {
    return ParallelSorter<ExtendedConfig<It, Comp, Cfg, std::decay_t<ThreadPool>>>(
        std::move(comp), std::forward<ThreadPool>(thread_pool));
}

template <class It, class Cfg = Config<>, class Comp = std::less<>>
ParallelSorter<ExtendedConfig<It, Comp, Cfg>>
make_sorter(int num_threads = DefaultThreadPool::maxNumThreads(), Comp comp = Comp()) {
    return ParallelSorter<ExtendedConfig<It, Comp, Cfg>>(std::move(comp),
                                                         DefaultThreadPool(num_threads));
}

template <class Cfg = Config<>, class It, class Comp, class ThreadPool>
void sort(It begin, It end, Comp comp, ThreadPool&& thread_pool) {
    if (Cfg::numThreadsFor(begin, end, thread_pool.numThreads()) < 2)
        ips4o::sort<Cfg>(std::move(begin), std::move(end), std::move(comp));
    else
        make_sorter<It, Cfg>(std::forward<ThreadPool>(thread_pool),
                             std::move(comp))(std::move(begin), std::move(end));
}

template <class Cfg = Config<>, class It, class Comp>
void sort(It begin, It end, Comp comp, int num_threads) {
    num_threads = Cfg::numThreadsFor(begin, end, num_threads);
    if (num_threads < 2)
        ips4o::sort<Cfg>(std::move(begin), std::move(end), std::move(comp));
    else
        make_sorter<It, Cfg>(num_threads, std::move(comp))(std::move(begin), std::move(end));
}

template <class It, class Comp> void sort(It begin, It end, Comp comp) {
    parallel::sort<Config<>>(std::move(begin), std::move(end), std::move(comp),
                             DefaultThreadPool::maxNumThreads());
}

template <class It> void sort(It begin, It end) {
    parallel::sort(std::move(begin), std::move(end), std::less<>());
}

} // namespace parallel
} // namespace algoat::sorting::detail::ips4o
