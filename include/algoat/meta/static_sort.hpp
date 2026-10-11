/**
 * @file static_sort.hpp
 * @brief Zero-overhead compile-time algorithm selector and static sort interface.
 *
 * Implements compile-time policy-driven algorithm selection with strict O(0) dispatch
 * overhead, complete inlining, and dead-code elimination.
 */

#pragma once

#include "algoat/meta/policy.hpp"
#include "algoat/meta/sorting_network.hpp"
#include "algoat/sorting/blocksort.hpp"
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/radixsort.hpp"
#include "algoat/sorting/timsort.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>

namespace algoat::meta {

namespace detail {

/**
 * @brief Core static algorithm dispatch engine.
 *
 * Evaluates PolicyDescriptor and span traits at compile time to inline the optimal
 * sorting kernel without any runtime overhead, vtables, or dynamic jump tables.
 *
 * @tparam Desc The compile-time PolicyDescriptor.
 * @tparam T Element type.
 * @tparam N Span extent.
 * @tparam Compare Strict weak ordering comparator.
 */
template <PolicyDescriptor Desc, typename T, std::size_t N, typename Compare = std::less<T>>
constexpr void dispatch_static_sort(std::span<T, N> data, Compare comp = {}) {
    // 1. Trivial base case: 0 or 1 element is guaranteed sorted at compile time
    if constexpr (N != std::dynamic_extent && N <= 1) {
        return;
    } else {
        if (data.size() <= 1) {
            return;
        }

        // 2. Forced Radix Sort kernel validation and execution
        if constexpr (Desc.force_radix) {
            static_assert(std::is_integral_v<T>,
                          "algoat::meta::static_sort: force_radix policy is only supported for "
                          "integral types.");
            static_assert(!std::is_same_v<std::remove_cvref_t<T>, bool>,
                          "algoat::meta::static_sort: force_radix policy is not applicable to "
                          "bool.");
            static_assert(std::is_same_v<Compare, std::less<T>>,
                          "algoat::meta::static_sort: non-comparative radix sort cannot accept a "
                          "custom comparator.");

            if constexpr (Desc.stable) {
                // Stable Radix sort requires out-of-place buffer passes
                algoat::sorting::radixsort_lsd(std::span<T>{data});
            } else if constexpr (Desc.in_place) {
                // In-place MSD Radix sort is efficient but unstable
                algoat::sorting::radixsort_inplace_msd(std::span<T>{data});
            } else {
                algoat::sorting::radixsort_lsd(std::span<T>{data});
            }
            return;
        }

        // 3. Statically-known small extents (N <= 16): optimal unrolled sorting network
        if constexpr (N != std::dynamic_extent && N <= 16 && !Desc.stable) {
            algoat::meta::sort_network<T, N, Compare>(data, comp);
            return;
        }

        // 4. Statically-known extents (N <= 32): zero-overhead insertion sort
        if constexpr (N != std::dynamic_extent && N <= 32) {
            algoat::sorting::insertionsort(std::span<T>{data}, comp);
            return;
        }

        // 5. Adaptive nearly-sorted hint -> TimSort (or InsertionSort for small spans)
        if constexpr (Desc.nearly_sorted) {
            if constexpr (Desc.in_place && N != std::dynamic_extent && N <= 64) {
                algoat::sorting::insertionsort(std::span<T>{data}, comp);
            } else {
                algoat::sorting::timsort(std::span<T>{data}, comp);
            }
            return;
        }

        // 6. Strict stability guarantees
        if constexpr (Desc.stable) {
            if constexpr (Desc.in_place) {
                algoat::sorting::blocksort(std::span<T>{data}, comp);
            } else {
                algoat::sorting::mergesort(std::span<T>{data}, comp);
            }
            return;
        }

        // 7. General default: introsort (O(N log N) hybrid quicksort / heapsort / insertionsort)
        algoat::sorting::introsort(std::span<T>{data}, comp);
    }
}

} // namespace detail

/**
 * @brief Zero-overhead compile-time static sort using C++20 structural class NTTP descriptor.
 *
 * @tparam Desc PolicyDescriptor structural NTTP specifying compile-time requirements.
 * @tparam T Element type.
 * @tparam N Span extent.
 * @tparam Compare Strict weak ordering comparator callable.
 *
 * @param data Contiguous span of elements to sort in-place.
 * @param comp Strict weak ordering comparator.
 */
template <PolicyDescriptor Desc, typename T, std::size_t N = std::dynamic_extent,
          typename Compare = std::less<T>>
constexpr void static_sort(std::span<T, N> data, Compare comp = {}) {
    detail::dispatch_static_sort<Desc>(data, comp);
}

/**
 * @brief Zero-overhead compile-time static sort using declarative policy tag chains.
 *
 * Matches the canonical issue blueprint:
 * template <typename Policy = PolicyChain<>, typename T, std::size_t N = std::dynamic_extent>
 * constexpr void static_sort(std::span<T, N> data, Policy = {});
 *
 * Deduces the policy tag chain at compile time and forwards directly to the
 * structural NTTP implementation with zero runtime dispatch cost.
 *
 * @tparam Policy PolicyChain or policy tag type.
 * @tparam T Element type.
 * @tparam N Span extent.
 *
 * @param data Contiguous span of elements to sort in-place.
 * @param policy Declarative policy chain or tag (e.g. p::stable | p::in_place). Defaults to
 * PolicyChain<>{}.
 */
template <typename Policy = PolicyChain<>, typename T, std::size_t N = std::dynamic_extent>
    requires(!std::is_same_v<Policy, PolicyDescriptor>)
constexpr void static_sort(std::span<T, N> data, Policy policy = {}) {
    (void)policy;
    if constexpr (IsPolicyTag<Policy>) {
        constexpr PolicyDescriptor desc = Policy{};
        static_sort<desc>(data);
    } else {
        constexpr PolicyDescriptor desc = Policy::to_descriptor();
        static_sort<desc>(data);
    }
}

} // namespace algoat::meta
