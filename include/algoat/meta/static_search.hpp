/**
 * @file static_search.hpp
 * @brief Zero-overhead NTTP static policy dispatch for searching algorithms.
 *
 * Translates a structural SearchPolicyDescriptor or declarative PolicyChain
 * into a compile-time `if constexpr` specialization with zero runtime branching.
 */

#pragma once

#include "algoat/meta/policy.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/searching/eytzinger_search_tree.hpp"
#include "algoat/searching/galloping_search.hpp"
#include "algoat/searching/searching.hpp"
#include "algoat/searching/static_btree.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::meta {

namespace detail {

/**
 * @brief Builds and queries a layout-reorganizing search tree (Eytzinger or Static B-Tree).
 */
template <bool UseStaticBTree, bool Prefetch, std::ranges::random_access_range R, typename T>
[[nodiscard]] std::optional<std::size_t> search_with_layout_tree(R&& data, const T& target) {
    using ValueType = std::ranges::range_value_t<R>;
    using RawT = std::remove_cvref_t<T>;
    const std::size_t n = std::ranges::size(data);
    if (n == 0) {
        return std::nullopt;
    }

    auto query_span = [&](std::span<const ValueType> span,
                          const ValueType& key) -> std::optional<std::size_t> {
        std::size_t idx = n;
        if constexpr (UseStaticBTree) {
            algoat::searching::StaticBTree<ValueType> tree(span);
            idx = tree.template lower_bound<Prefetch>(key);
        } else {
            algoat::searching::EytzingerSearchTree<ValueType> tree(span);
            idx = tree.template lower_bound<Prefetch>(key);
        }
        if (idx < n && !(key < span[idx]) && !(span[idx] < key)) {
            return idx;
        }
        return std::nullopt;
    };

    // Only convert heterogeneous keys when conversion cannot alter value or signedness.
    constexpr bool same_value = std::same_as<ValueType, RawT>;
    constexpr bool safe_non_arithmetic = !std::is_arithmetic_v<ValueType> &&
                                         !std::is_arithmetic_v<RawT> &&
                                         std::convertible_to<const T&, ValueType>;
    constexpr bool safe_integral = std::is_integral_v<ValueType> && std::is_integral_v<RawT> &&
                                   !std::is_same_v<ValueType, bool> &&
                                   !std::is_same_v<RawT, bool> &&
                                   (std::is_signed_v<ValueType> == std::is_signed_v<RawT>) &&
                                   (sizeof(RawT) <= sizeof(ValueType));

    if constexpr (!(same_value || safe_non_arithmetic || safe_integral)) {
        return algoat::searching::binary_search(data, target);
    } else {
        auto dispatch_key = [&](std::span<const ValueType> span) -> std::optional<std::size_t> {
            if constexpr (same_value) {
                return query_span(span, target);
            } else {
                const ValueType key = static_cast<ValueType>(target);
                return query_span(span, key);
            }
        };

        if constexpr (std::ranges::contiguous_range<R>) {
            return dispatch_key(std::span<const ValueType>{std::ranges::data(data), n});
        } else {
            std::vector<ValueType> contiguous(std::ranges::begin(data), std::ranges::end(data));
            return dispatch_key(std::span<const ValueType>{contiguous});
        }
    }
}

/**
 * @brief Compile-time static policy dispatcher for searching kernels.
 */
template <SearchPolicyDescriptor Policy, typename R, typename T>
[[nodiscard]] constexpr std::optional<std::size_t> dispatch_static_search(R&& data,
                                                                          const T& target) {
    static_assert(std::ranges::random_access_range<R>,
                  "static_search requires a std::ranges::random_access_range (such as "
                  "std::vector, std::array, std::span, or std::deque).");
    static_assert(!(Policy.eytzinger && Policy.static_btree),
                  "Search policy conflict: p::eytzinger and p::static_btree are mutually "
                  "exclusive memory layouts.");
    static_assert(!(Policy.galloping && (Policy.eytzinger || Policy.static_btree)),
                  "Search policy conflict: p::galloping cannot be combined with p::eytzinger or "
                  "p::static_btree.");
    static_assert(!(Policy.branchless && Policy.galloping),
                  "Search policy conflict: p::branchless cannot be combined with p::galloping.");
    static_assert(!(Policy.branchless && Policy.static_btree),
                  "Search policy conflict: p::branchless cannot be combined with p::static_btree.");

    using ValueType = std::ranges::range_value_t<R>;
    if constexpr (Policy.eytzinger || Policy.static_btree) {
        static_assert(std::totally_ordered<ValueType>,
                      "Layout-reorganizing search policies (p::eytzinger, p::static_btree) require "
                      "std::totally_ordered elements.");
    }

    if (std::ranges::empty(data)) {
        return std::nullopt;
    }

    if constexpr (Policy.static_btree) {
        if (std::is_constant_evaluated()) {
            return algoat::searching::binary_search(data, target);
        }
        return search_with_layout_tree<true, Policy.prefetch>(std::forward<R>(data), target);
    } else if constexpr (Policy.eytzinger) {
        if (std::is_constant_evaluated()) {
            return algoat::searching::binary_search(data, target);
        }
        return search_with_layout_tree<false, Policy.prefetch>(std::forward<R>(data), target);
    } else if constexpr (Policy.galloping) {
        auto first = std::ranges::begin(data);
        auto last = std::ranges::end(data);
        auto it = algoat::searching::galloping_search(first, last, target);
        return algoat::searching::detail::match_lower_bound_index(
            first, it, last, target, std::ranges::less{}, std::identity{});
    } else if constexpr (Policy.branchless) {
        auto first = std::ranges::begin(data);
        auto last = std::ranges::end(data);
        auto it = algoat::searching::branchless_lower_bound<Policy.prefetch>(first, last, target);
        return algoat::searching::detail::match_lower_bound_index(
            first, it, last, target, std::ranges::less{}, std::identity{});
    } else {
        return algoat::searching::binary_search(data, target);
    }
}

} // namespace detail

/**
 * @brief Zero-overhead static policy search using a compile-time NTTP SearchPolicyDescriptor.
 *
 * @tparam Policy Structural SearchPolicyDescriptor specifying search kernel and hardware hints.
 * @tparam R Random-access range type of sorted elements.
 * @tparam T Key type to locate.
 * @param data Sorted random-access range to search.
 * @param target Search key value.
 * @return Optional 0-based index of an element equal to target, or std::nullopt if not found.
 */
template <SearchPolicyDescriptor Policy = SearchPolicyDescriptor{},
          std::ranges::random_access_range R, typename T>
[[nodiscard]] constexpr std::optional<std::size_t> static_search(R&& data, const T& target) {
    return detail::dispatch_static_search<Policy>(std::forward<R>(data), target);
}

/**
 * @brief Declarative PolicyChain overload for zero-overhead static policy search.
 *
 * @tparam R Range type of sorted elements (checked via static_assert for random_access_range).
 * @tparam T Key type to locate.
 * @tparam Tags Variadic sequence of policy tag types.
 * @param data Sorted random-access range to search.
 * @param target Search key value.
 * @param chain Declarative policy chain composed with operator|.
 * @return Optional 0-based index of an element equal to target, or std::nullopt if not found.
 */
template <typename R, typename T, typename... Tags>
    requires(IsPolicyTag<Tags> && ...)
[[nodiscard]] constexpr std::optional<std::size_t> static_search(R&& data, const T& target,
                                                                 PolicyChain<Tags...> /*chain*/) {
    constexpr SearchPolicyDescriptor desc = PolicyChain<Tags...>::to_search_descriptor();
    return detail::dispatch_static_search<desc>(std::forward<R>(data), target);
}

/**
 * @brief Convenience single-tag overload for static_search.
 *
 * @tparam R Range type of sorted elements (checked via static_assert for random_access_range).
 * @tparam T Key type to locate.
 * @tparam Tag Single policy tag satisfying IsPolicyTag.
 * @param data Sorted random-access range to search.
 * @param target Search key value.
 * @param tag Policy tag.
 * @return Optional 0-based index of an element equal to target, or std::nullopt if not found.
 */
template <typename R, typename T, IsPolicyTag Tag>
[[nodiscard]] constexpr std::optional<std::size_t> static_search(R&& data, const T& target,
                                                                 Tag /*tag*/) {
    constexpr SearchPolicyDescriptor desc = PolicyChain<Tag>::to_search_descriptor();
    return detail::dispatch_static_search<desc>(std::forward<R>(data), target);
}

} // namespace algoat::meta
