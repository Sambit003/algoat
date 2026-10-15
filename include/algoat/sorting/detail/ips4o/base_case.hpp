/******************************************************************************
 * algoat/sorting/detail/ips4o/base_case.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "algoat/sorting/pdqsort.hpp"
#include "ips4o_fwd.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class It, class Comp> void insertionSort(const It begin, const It end, Comp comp) {
    if (begin >= end)
        return;

    for (It it = begin + 1; it < end; ++it) {
        typename std::iterator_traits<It>::value_type val = std::move(*it);
        if (comp(val, *begin)) {
            std::move_backward(begin, it, it + 1);
            *begin = std::move(val);
        } else {
            auto cur = it;
            for (auto next = it - 1; comp(val, *next); --next) {
                *cur = std::move(*next);
                cur = next;
            }
            *cur = std::move(val);
        }
    }
}

template <class It, class Comp> inline void baseCaseSort(It begin, It end, Comp&& comp) {
    if (begin == end)
        return;
    if (end - begin <= 16) {
        detail::insertionSort(begin, end, std::forward<Comp>(comp));
    } else {
        ::algoat::sorting::pdqsort(begin, end, std::forward<Comp>(comp));
    }
}

template <class It, class Comp> inline bool sortedCaseSort(It begin, It end, Comp&& comp) {
    if (begin == end) {
        return true;
    }

    if (!comp(*(end - 1), *begin)) {
        if (std::is_sorted(begin, end, comp)) {
            return true;
        }
    } else {
        for (It it = begin; (it + 1) != end; ++it) {
            if (comp(*it, *(it + 1))) {
                return false;
            }
        }
        std::reverse(begin, end);
        return true;
    }

    return false;
}

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
