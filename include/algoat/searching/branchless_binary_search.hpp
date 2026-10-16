/**
 * @file branchless_binary_search.hpp
 * @brief Branchless binary search (lower_bound) utilizing conditional moves (CMOV/CSEL).
 */

#pragma once

#include "algoat/core/registry.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::searching {

namespace detail {

/**
 * @brief Conditionally advances an iterator using a conditional move instruction when possible.
 *
 * Emits CMOV on x86-64 and CSEL on ARM64 for pointer iterators outside constant evaluation,
 * completely avoiding conditional branches.
 */
template <typename I>
[[nodiscard]] constexpr I cmov_advance(I base, std::iter_difference_t<I> step, bool cond) noexcept {
    if (std::is_constant_evaluated()) {
        return cond ? (base + step) : base;
    } else {
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__ILP32__)
        if constexpr (std::is_pointer_v<I>) {
            I res = base;
            I alt = base + step;
            asm("testb %1, %1\n\tcmovnzq %2, %0" : "+r"(res) : "r"(cond), "r"(alt) : "cc");
            return res;
        } else {
            return cond ? (base + step) : base;
        }
#elif defined(__aarch64__)
        if constexpr (std::is_pointer_v<I>) {
            I res;
            I alt = base + step;
            asm("cmp %w2, #0\n\tcsel %0, %1, %3, ne"
                : "=r"(res)
                : "r"(alt), "r"(cond), "r"(base)
                : "cc");
            return res;
        } else {
            return cond ? (base + step) : base;
        }
#else
        return cond ? (base + step) : base;
#endif
    }
}

/**
 * @brief Core branchless binary search loop operating on any random-access iterator.
 */
template <typename I, typename S, typename T, typename Comp, typename Proj>
[[nodiscard]] constexpr I branchless_search_core(I first, S last, const T& target, Comp comp,
                                                 Proj proj) {
    using diff_t = std::iter_difference_t<I>;
    diff_t len = last - first;
    I base = first;

    while (len > 1) {
        const diff_t half = len >> 1;
        const bool take_right =
            bool(std::invoke(comp, std::invoke(proj, *(base + (half - 1))), target));
        base = detail::cmov_advance(base, half, take_right);
        len -= half;
    }

    if (base < last && bool(std::invoke(comp, std::invoke(proj, *base), target))) {
        ++base;
    }

    return base;
}

} // namespace detail

/**
 * @brief Finds the first element in [first, last) not satisfying comp(proj(*it), target).
 *
 * Implements branchless binary search with lower_bound semantics. The algorithm eliminates
 * branch mispredictions by executing a fixed number of ceiling-halved iterations and updating
 * the base pointer via conditional moves (CMOV on x86-64, CSEL on AArch64).
 *
 * @tparam I Random access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam T Target value type.
 * @tparam Proj Projection callable.
 * @tparam Comp Strict weak ordering comparator callable.
 *
 * @param first Beginning of range.
 * @param last End / sentinel of range.
 * @param target Value to locate lower bound for.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Iterator pointing to the first element not less than target, or @c last if all elements
 * are less than target.
 */
template <
    std::random_access_iterator I, std::sentinel_for<I> S, typename T,
    typename Proj = std::identity,
    std::indirect_strict_weak_order<const T*, std::projected<I, Proj>> Comp = std::ranges::less>
[[nodiscard]] constexpr I branchless_lower_bound(I first, S last, const T& target, Comp comp = {},
                                                 Proj proj = {}) {
    if constexpr (std::contiguous_iterator<I> && std::contiguous_iterator<S>) {
        if (!std::is_constant_evaluated()) {
            const auto* first_ptr = std::to_address(first);
            const auto* last_ptr = std::to_address(last);
            const auto* res_ptr =
                detail::branchless_search_core(first_ptr, last_ptr, target, comp, proj);
            return first + (res_ptr - first_ptr);
        }
    }

    return detail::branchless_search_core(first, last, target, comp, proj);
}

/**
 * @brief Overload of @c branchless_lower_bound accepting a C++20 random access range.
 *
 * @tparam R Random-access range type.
 * @tparam T Target value type.
 * @tparam Proj Projection callable.
 * @tparam Comp Strict weak ordering comparator callable.
 *
 * @param range The sorted range to search.
 * @param target Target value.
 * @param comp Strict weak ordering comparator.
 * @param proj Element projection callable.
 * @return Borrowed iterator pointing to the lower bound, or @c std::ranges::end(range).
 */
template <
    std::ranges::random_access_range R, typename T, typename Proj = std::identity,
    std::indirect_strict_weak_order<const T*, std::projected<std::ranges::iterator_t<R>, Proj>>
        Comp = std::ranges::less>
[[nodiscard]] constexpr std::ranges::borrowed_iterator_t<R>
branchless_lower_bound(R&& range, const T& target, Comp comp = {}, Proj proj = {}) {
    return branchless_lower_bound(std::ranges::begin(range), std::ranges::end(range), target,
                                  std::move(comp), std::move(proj));
}

/**
 * @struct BranchlessBinarySearch
 * @brief Search algorithm adapter wrapping @c branchless_lower_bound to satisfy @c SearchAlgorithm.
 */
struct BranchlessBinarySearch {
    /**
     * @brief Returns unique algorithm identifier.
     * @return "branchlessbinarysearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "branchlessbinarysearch";
    }

    /**
     * @brief Indicates whether sorted input is required.
     * @return @c true
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }

    /**
     * @brief Searches for target within a const span using branchless lower bound.
     *
     * @tparam T Element type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T>
    constexpr std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        return search(data, target, std::ranges::less{});
    }

    /**
     * @brief Searches for target within a mutable span using branchless lower bound.
     *
     * @tparam T Element type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T>
    constexpr std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target, std::ranges::less{});
    }

    /**
     * @brief Searches for target within a const span using a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @param comp Strict weak ordering comparator.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T, typename Compare>
    constexpr std::optional<std::size_t> search(std::span<const T> data, const T& target,
                                                Compare comp) const {
        if (data.empty()) {
            return std::nullopt;
        }
        auto it = branchless_lower_bound(data.begin(), data.end(), target, comp);
        if (it != data.end() && !comp(*it, target) && !comp(target, *it)) {
            return static_cast<std::size_t>(it - data.begin());
        }
        return std::nullopt;
    }

    /**
     * @brief Searches for target within a mutable span using a custom comparator.
     *
     * @tparam T Element type.
     * @tparam Compare Strict weak ordering comparator type.
     * @param data Sorted span of elements.
     * @param target Value to locate.
     * @param comp Strict weak ordering comparator.
     * @return Index of matching element if present, or @c std::nullopt.
     */
    template <typename T, typename Compare>
    constexpr std::optional<std::size_t> search(std::span<T> data, const T& target,
                                                Compare comp) const {
        return search(std::span<const T>{data.data(), data.size()}, target, std::move(comp));
    }
};

} // namespace algoat::searching

ALGOAT_REGISTER_ALGORITHM("searching", "branchlessbinarysearch",
                          ::algoat::searching::BranchlessBinarySearch)
