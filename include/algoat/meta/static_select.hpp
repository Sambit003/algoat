/**
 * @file static_select.hpp
 * @brief Zero-overhead NTTP static policy dispatch for selection algorithms.
 *
 * Translates a structural SelectPolicyDescriptor or declarative PolicyChain
 * into a compile-time `if constexpr` specialization with zero runtime branching.
 */

#pragma once

#include "algoat/meta/policy.hpp"
#include "algoat/selection/nth_element.hpp"
#include "algoat/selection/pdqselect.hpp"
#include "algoat/selection/select_optimal.hpp"

#include <cstddef>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <utility>

namespace algoat::meta {

namespace detail {

/**
 * @brief Compile-time static policy dispatcher for selection kernels.
 */
template <SelectPolicyDescriptor Policy, typename R>
[[nodiscard]] constexpr std::ranges::iterator_t<R> dispatch_static_select(R&& data, std::size_t k) {
    static_assert(std::ranges::random_access_range<R>,
                  "static_select requires a std::ranges::random_access_range (such as "
                  "std::vector, std::array, std::span, or std::deque).");
    static_assert(std::permutable<std::ranges::iterator_t<R>>,
                  "static_select requires a mutable random-access range whose elements are "
                  "swappable.");
    static_assert(Policy.in_place,
                  "Selection policy conflict: static_select mutates elements in place and "
                  "requires in_place = true.");
    static_assert(!(Policy.floyd_rivest && Policy.branchless),
                  "Selection policy conflict: p::floyd_rivest and p::branchless are mutually "
                  "exclusive selection kernels.");

    const auto n = std::ranges::distance(data);
    if (n <= 0 || k >= static_cast<std::size_t>(n)) {
        return std::ranges::end(data);
    }

    auto first = std::ranges::begin(data);
    auto last = std::ranges::end(data);
    auto nth = first + static_cast<std::iter_difference_t<decltype(first)>>(k);

    if constexpr (Policy.floyd_rivest) {
        algoat::selection::select_optimal(first, nth, last);
        return nth;
    } else if constexpr (Policy.branchless) {
        if (std::is_constant_evaluated()) {
            algoat::selection::nth_element(first, nth, last);
            return nth;
        }
        algoat::selection::pdqselect(first, nth, last);
        return nth;
    } else {
        algoat::selection::nth_element(first, nth, last);
        return nth;
    }
}

} // namespace detail

/**
 * @brief Zero-overhead static policy selection using a compile-time NTTP SelectPolicyDescriptor.
 *
 * @tparam Policy Structural SelectPolicyDescriptor selecting kernel and complexity guarantees.
 * @tparam R Mutable random-access range type.
 * @param data Mutable random-access range to partially order in place.
 * @param k Zero-based rank of the desired element.
 * @return Iterator to the partitioned kth element, or std::ranges::end(data) if k >= size.
 */
template <SelectPolicyDescriptor Policy = SelectPolicyDescriptor{},
          std::ranges::random_access_range R>
[[nodiscard]] constexpr std::ranges::iterator_t<R> static_select(R&& data, std::size_t k) {
    return detail::dispatch_static_select<Policy>(std::forward<R>(data), k);
}

/**
 * @brief Declarative PolicyChain overload for zero-overhead static policy selection.
 *
 * @tparam R Mutable range type (checked via static_assert for random_access_range).
 * @tparam Tags Variadic sequence of policy tag types.
 * @param data Mutable random-access range to partially order in place.
 * @param k Zero-based rank of the desired element.
 * @param chain Declarative policy chain composed with operator|.
 * @return Iterator to the partitioned kth element, or std::ranges::end(data) if k >= size.
 */
template <typename R, typename... Tags>
    requires(IsPolicyTag<Tags> && ...)
[[nodiscard]] constexpr std::ranges::iterator_t<R> static_select(R&& data, std::size_t k,
                                                                 PolicyChain<Tags...> /*chain*/) {
    constexpr SelectPolicyDescriptor desc = PolicyChain<Tags...>::to_select_descriptor();
    return detail::dispatch_static_select<desc>(std::forward<R>(data), k);
}

/**
 * @brief Convenience single-tag overload for static_select.
 *
 * @tparam R Mutable range type (checked via static_assert for random_access_range).
 * @tparam Tag Single policy tag satisfying IsPolicyTag.
 * @param data Mutable random-access range to partially order in place.
 * @param k Zero-based rank of the desired element.
 * @param tag Policy tag.
 * @return Iterator to the partitioned kth element, or std::ranges::end(data) if k >= size.
 */
template <typename R, IsPolicyTag Tag>
[[nodiscard]] constexpr std::ranges::iterator_t<R> static_select(R&& data, std::size_t k,
                                                                 Tag /*tag*/) {
    constexpr SelectPolicyDescriptor desc = PolicyChain<Tag>::to_select_descriptor();
    return detail::dispatch_static_select<desc>(std::forward<R>(data), k);
}

} // namespace algoat::meta
