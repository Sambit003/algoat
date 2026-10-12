/**
 * @file fluxsort.hpp
 * @brief High-performance Fluxsort sorting engine implementation.
 *
 * Implements Igor van den Hoven's (scandum) Fluxsort algorithm: an adaptive, stable
 * hybrid sort combining top-down presortedness analysis, branchless stable flux partitioning,
 * reverse partitioning for duplicate keys, and Quadsort 4-way ping-pong branchless merges.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/sorting/detail/quadsort_engine.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::sorting {

namespace detail {

inline constexpr std::size_t flux_out_threshold = 96;

/**
 * @brief Exception-safe RAII scratch buffer for Fluxsort backed by std::allocator.
 */
template <typename T> struct FluxScratchBuffer {
    std::allocator<T> alloc_{};
    T* memory_{nullptr};
    std::size_t capacity_{0};
    std::size_t constructed_{0};

    explicit FluxScratchBuffer(std::size_t cap)
        : memory_(cap > 0 ? std::allocator_traits<std::allocator<T>>::allocate(alloc_, cap)
                          : nullptr),
          capacity_(cap) {
        if constexpr (!std::is_trivially_copyable_v<T>) {
            if (memory_ != nullptr) {
                try {
                    std::uninitialized_default_construct_n(memory_, capacity_);
                    constructed_ = capacity_;
                } catch (...) {
                    std::allocator_traits<std::allocator<T>>::deallocate(alloc_, memory_,
                                                                         capacity_);
                    memory_ = nullptr;
                    capacity_ = 0;
                    throw;
                }
            }
        }
    }

    ~FluxScratchBuffer() {
        if (memory_ != nullptr) {
            if constexpr (!std::is_trivially_copyable_v<T>) {
                std::destroy_n(memory_, constructed_);
            }
            std::allocator_traits<std::allocator<T>>::deallocate(alloc_, memory_, capacity_);
        }
    }

    FluxScratchBuffer(const FluxScratchBuffer&) = delete;
    FluxScratchBuffer& operator=(const FluxScratchBuffer&) = delete;

    FluxScratchBuffer(FluxScratchBuffer&& other) noexcept
        : alloc_(std::move(other.alloc_)), memory_(other.memory_), capacity_(other.capacity_),
          constructed_(other.constructed_) {
        other.memory_ = nullptr;
        other.capacity_ = 0;
        other.constructed_ = 0;
    }

    FluxScratchBuffer& operator=(FluxScratchBuffer&& other) noexcept {
        if (this != &other) {
            if (memory_ != nullptr) {
                if constexpr (!std::is_trivially_copyable_v<T>) {
                    std::destroy_n(memory_, constructed_);
                }
                std::allocator_traits<std::allocator<T>>::deallocate(alloc_, memory_, capacity_);
            }
            alloc_ = std::move(other.alloc_);
            memory_ = other.memory_;
            capacity_ = other.capacity_;
            constructed_ = other.constructed_;
            other.memory_ = nullptr;
            other.capacity_ = 0;
            other.constructed_ = 0;
        }
        return *this;
    }

    [[nodiscard]] T* data() noexcept {
        return memory_;
    }

    [[nodiscard]] const T* data() const noexcept {
        return memory_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return capacity_;
    }
};

/**
 * @brief Default stable flux partition.
 *
 * Partitions elements in [array, array + nmemb) stable: elements <= piv stay in array,
 * elements > piv are moved to swap, then copied to array + m.
 */
template <std::random_access_iterator I, typename T, typename Comp, typename Proj>
std::size_t flux_default_partition(I array, T* swap, std::size_t nmemb,
                                   const std::iter_value_t<I>& piv, Comp& comp, Proj& proj) {
    auto&& piv_key = std::invoke(proj, piv);

    std::size_t m = 0;
    std::size_t s = 0;

    for (std::size_t i = 0; i < nmemb; ++i) {
        bool val = !std::invoke(comp, piv_key, std::invoke(proj, array[i]));
        if (val) {
            if (m != i) {
                array[m] = std::ranges::iter_move(array + i);
            }
            ++m;
        } else {
            swap[s++] = std::ranges::iter_move(array + i);
        }
    }

    // Append swap elements to array + m
    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(std::to_address(array + m), std::to_address(swap), s * sizeof(T));
    } else {
        for (std::size_t i = 0; i < s; ++i) {
            array[m + i] = std::ranges::iter_move(swap + i);
        }
    }

    return m;
}

/**
 * @brief Reverse stable flux partition for duplicate key handling.
 *
 * Partitions elements with predicate (elem < piv) to isolate elements equal to piv.
 */
template <std::random_access_iterator I, typename T, typename Comp, typename Proj>
std::size_t flux_reverse_partition(I array, T* swap, std::size_t nmemb,
                                   const std::iter_value_t<I>& piv, Comp& comp, Proj& proj) {
    auto&& piv_key = std::invoke(proj, piv);

    std::size_t m = 0;
    std::size_t s = 0;

    for (std::size_t i = 0; i < nmemb; ++i) {
        bool val = std::invoke(comp, std::invoke(proj, array[i]), piv_key);
        if (val) {
            if (m != i) {
                array[m] = std::ranges::iter_move(array + i);
            }
            ++m;
        } else {
            swap[s++] = std::ranges::iter_move(array + i);
        }
    }

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(std::to_address(array + m), std::to_address(swap), s * sizeof(T));
    } else {
        for (std::size_t i = 0; i < s; ++i) {
            array[m + i] = std::ranges::iter_move(swap + i);
        }
    }

    return m;
}

/**
 * @brief Pivot selection for Fluxsort.
 */
template <std::random_access_iterator I, typename T, typename Comp, typename Proj>
std::iter_value_t<I> flux_select_pivot(I array, T* swap, std::size_t swap_size, std::size_t nmemb,
                                       bool& generic, Comp& comp, Proj& proj) {
    if (nmemb <= 2048) {
        I ptp = median_of_nine(array, nmemb, comp, proj);
        return *ptp;
    }

    std::size_t cbrt = 32;
    while (nmemb > cbrt * cbrt * cbrt && cbrt * 4 <= swap_size) {
        cbrt *= 2;
    }

    std::size_t div = nmemb / cbrt;
    std::size_t offset = div / 2;

    for (std::size_t cnt = 0; cnt < cbrt; ++cnt) {
        swap[cnt] = array[offset + cnt * div];
    }

    std::span<T> swap_scratch(swap + cbrt, swap_size - cbrt);
    std::size_t half = cbrt / 2;
    tail_swap(swap, swap_scratch, half, comp, proj);
    tail_swap(swap + half, swap_scratch, half, comp, proj);

    generic = !std::invoke(comp, std::invoke(proj, swap[0]), std::invoke(proj, swap[cbrt - 1])) &&
              !std::invoke(comp, std::invoke(proj, swap[cbrt - 1]), std::invoke(proj, swap[0]));

    T* result = binary_median(swap, swap + half, half, comp, proj);
    return *result;
}

/**
 * @brief Recursive stable flux partitioning orchestrator.
 */
template <std::random_access_iterator I, typename T, typename Comp, typename Proj>
void flux_partition(I array, T* swap, std::size_t swap_size, std::size_t nmemb, Comp& comp,
                    Proj& proj) {
    while (nmemb > flux_out_threshold) {
        bool generic = false;
        auto piv = flux_select_pivot(array, swap, swap_size, nmemb, generic, comp, proj);

        std::span<T> swap_span(swap, swap_size);
        if (generic) {
            tail_swap(array, swap_span, nmemb, comp, proj);
            return;
        }

        std::size_t a_size = flux_default_partition(array, swap, nmemb, piv, comp, proj);
        std::size_t s_size = nmemb - a_size;

        if (a_size <= s_size / 32 || s_size <= flux_out_threshold) {
            if (a_size == 0) {
                return;
            }
            if (s_size == 0) {
                a_size = flux_reverse_partition(array, swap, a_size, piv, comp, proj);
                if (a_size <= flux_out_threshold) {
                    tail_swap(array, swap_span, a_size, comp, proj);
                } else {
                    flux_partition(array, swap, swap_size, a_size, comp, proj);
                }
                return;
            }
            tail_swap(array + a_size, swap_span, s_size, comp, proj);
        } else {
            flux_partition(array + a_size, swap, swap_size, s_size, comp, proj);
        }

        if (s_size <= a_size / 32 || a_size <= flux_out_threshold) {
            if (a_size <= flux_out_threshold) {
                tail_swap(array, swap_span, a_size, comp, proj);
            } else {
                a_size = flux_reverse_partition(array, swap, a_size, piv, comp, proj);
                if (a_size <= flux_out_threshold) {
                    tail_swap(array, swap_span, a_size, comp, proj);
                } else {
                    flux_partition(array, swap, swap_size, a_size, comp, proj);
                }
            }
            return;
        }

        nmemb = a_size;
    }

    if (nmemb > 1) {
        std::span<T> swap_span(swap, swap_size);
        tail_swap(array, swap_span, nmemb, comp, proj);
    }
}

} // namespace detail

/**
 * @brief Sorts elements in the range [first, last) stably using Fluxsort.
 *
 * Fluxsort is an adaptive, stable hybrid sort that combines top-down presortedness
 * analysis, branchless stable flux partitioning, and Quadsort 4-way merge networks.
 *
 * @tparam I Random access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable.
 *
 * @param first Iterator pointing to the beginning of the range.
 * @param last Sentinel pointing to the end of the range.
 * @param comp Strict weak ordering comparator callable.
 * @param proj Projection callable.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
void fluxsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto end_it = std::ranges::next(first, last);
    auto count = static_cast<std::size_t>(end_it - first);

    if (count <= 1) {
        return;
    }

    using ValueType = std::iter_value_t<I>;

    if (count <= detail::flux_out_threshold) {
        std::array<ValueType, detail::flux_out_threshold> stack_swap;
        detail::tail_swap(first, stack_swap, count, comp, proj);
        return;
    }

    auto presorted = detail::analyze_presortedness(first, count, comp, proj);
    if (presorted == detail::PresortedResult::AlreadySorted) {
        return;
    }

    detail::FluxScratchBuffer<ValueType> scratch(count);
    detail::flux_partition(first, scratch.data(), count, count, comp, proj);
}

/**
 * @brief Sorts a C++20 random access range stably using Fluxsort.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
void fluxsort(R&& r, Comp comp = {}, Proj proj = {}) {
    fluxsort(std::ranges::begin(r), std::ranges::end(r), std::move(comp), std::move(proj));
}

/**
 * @brief Sorts a contiguous span of elements stably using Fluxsort.
 */
template <typename T, typename Compare = std::ranges::less>
void fluxsort(std::span<T> data, Compare comp = {}) {
    fluxsort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @struct FluxSort
 * @brief Zero-overhead adapter struct for Fluxsort.
 *
 * Satisfies the @c algoat::sorting::SortAlgorithm concept for static polymorphism
 * and registry integration.
 */
struct FluxSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "fluxsort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    template <typename T> void sort(std::span<T> data) const {
        fluxsort(data);
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        fluxsort(data, std::move(comp));
    }
};

using Fluxsort = FluxSort;

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "fluxsort", ::algoat::sorting::FluxSort)
