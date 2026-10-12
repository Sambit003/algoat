/**
 * @file crumsort.hpp
 * @brief High-performance Crumsort sorting engine implementation.
 *
 * Implements Igor van den Hoven's (scandum) Crumsort algorithm: an unstable, branchless
 * in-place hybrid quicksort utilizing fulcrum partitioning, top-down presortedness analysis,
 * square-root pivot selection, and Quadsort small-sort fallbacks.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/sorting/detail/quadsort_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace algoat::sorting {

namespace detail {

inline constexpr std::size_t crum_out_threshold = 96;
inline constexpr std::size_t crum_aux_capacity = 512;

/**
 * @brief RAII guard to safely restore evacuated elements during fulcrum partitioning upon
 * exception.
 */
template <typename I, typename T> struct FulcrumEvacuationGuard {
    I left_dest;
    I right_dest;
    T* swap_buf;
    bool active{true};

    FulcrumEvacuationGuard(I l, I r, T* buf) : left_dest(l), right_dest(r), swap_buf(buf) {}

    ~FulcrumEvacuationGuard() {
        if (active && swap_buf != nullptr) {
            for (std::size_t i = 0; i < 32; ++i) {
                left_dest[i] = std::move(swap_buf[i]);
                right_dest[i] = std::move(swap_buf[32 + i]);
            }
        }
    }

    void dismiss() noexcept {
        active = false;
    }
};

/**
 * @brief Branchless default fulcrum partition with unrolled memcpy staging for trivially copyable
 * types.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
std::size_t fulcrum_default_partition(I array, std::size_t nmemb, const std::iter_value_t<I>& piv,
                                      const Comp& comp, const Proj& proj) {
    using T = std::iter_value_t<I>;
    std::array<T, 64> swap;

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(swap.data(), std::to_address(array), 32 * sizeof(T));
        std::memcpy(swap.data() + 32, std::to_address(array + nmemb - 32), 32 * sizeof(T));
    } else {
        for (std::size_t i = 0; i < 32; ++i) {
            swap[i] = std::ranges::iter_move(array + i);
            swap[32 + i] = std::ranges::iter_move(array + nmemb - 32 + i);
        }
    }

    FulcrumEvacuationGuard guard(array, array + nmemb - 32, swap.data());

    auto&& piv_key = std::invoke(proj, piv);

    I ptl = array;
    I ptr = array + nmemb - 1;
    I pta = array + 32;
    I tpa = array + nmemb - 33;
    std::size_t m = 0;
    std::size_t cnt = (nmemb / 16 >= 4) ? (nmemb / 16 - 4) : 0;

    while (cnt > 0) {
        if (static_cast<std::size_t>(pta - ptl) - m <= 48) {
            --cnt;
            for (std::size_t i = 0; i < 16; ++i) {
                bool val = !std::invoke(comp, piv_key, std::invoke(proj, *pta));
                if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                    ptl[m] = *pta;
                    ptr[m] = *pta++;
                } else {
                    if (val) {
                        ptl[m] = std::ranges::iter_move(pta++);
                    } else {
                        ptr[m] = std::ranges::iter_move(pta++);
                    }
                }
                m += val;
                --ptr;
            }
        }
        if (cnt == 0) {
            break;
        }
        if (static_cast<std::size_t>(pta - ptl) - m >= 16) {
            --cnt;
            for (std::size_t i = 0; i < 16; ++i) {
                bool val = !std::invoke(comp, piv_key, std::invoke(proj, *tpa));
                if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                    ptl[m] = *tpa;
                    ptr[m] = *tpa--;
                } else {
                    if (val) {
                        ptl[m] = std::ranges::iter_move(tpa--);
                    } else {
                        ptr[m] = std::ranges::iter_move(tpa--);
                    }
                }
                m += val;
                --ptr;
            }
        }
    }

    std::size_t rem = nmemb % 16;
    if (static_cast<std::size_t>(pta - ptl) - m <= 48) {
        while (rem-- > 0) {
            bool val = !std::invoke(comp, piv_key, std::invoke(proj, *pta));
            if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                ptl[m] = *pta;
                ptr[m] = *pta++;
            } else {
                if (val) {
                    ptl[m] = std::ranges::iter_move(pta++);
                } else {
                    ptr[m] = std::ranges::iter_move(pta++);
                }
            }
            m += val;
            --ptr;
        }
    } else {
        while (rem-- > 0) {
            bool val = !std::invoke(comp, piv_key, std::invoke(proj, *tpa));
            if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                ptl[m] = *tpa;
                ptr[m] = *tpa--;
            } else {
                if (val) {
                    ptl[m] = std::ranges::iter_move(tpa--);
                } else {
                    ptr[m] = std::ranges::iter_move(tpa--);
                }
            }
            m += val;
            --ptr;
        }
    }

    // Stream evacuated elements into central meeting hole
    for (std::size_t i = 0; i < 64; ++i) {
        bool val = !std::invoke(comp, piv_key, std::invoke(proj, swap[i]));
        if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
            ptl[m] = swap[i];
            ptr[m] = swap[i];
        } else {
            if (val) {
                ptl[m] = std::move(swap[i]);
            } else {
                ptr[m] = std::move(swap[i]);
            }
        }
        m += val;
        --ptr;
    }

    guard.dismiss();
    return m;
}

/**
 * @brief Reverse fulcrum partition for duplicate key handling.
 */
template <std::random_access_iterator I, typename Comp, typename Proj>
std::size_t fulcrum_reverse_partition(I array, std::size_t nmemb, const std::iter_value_t<I>& piv,
                                      const Comp& comp, const Proj& proj) {
    using T = std::iter_value_t<I>;
    std::array<T, 64> swap;

    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(swap.data(), std::to_address(array), 32 * sizeof(T));
        std::memcpy(swap.data() + 32, std::to_address(array + nmemb - 32), 32 * sizeof(T));
    } else {
        for (std::size_t i = 0; i < 32; ++i) {
            swap[i] = std::ranges::iter_move(array + i);
            swap[32 + i] = std::ranges::iter_move(array + nmemb - 32 + i);
        }
    }

    FulcrumEvacuationGuard guard(array, array + nmemb - 32, swap.data());

    auto&& piv_key = std::invoke(proj, piv);

    I ptl = array;
    I ptr = array + nmemb - 1;
    I pta = array + 32;
    I tpa = array + nmemb - 33;
    std::size_t m = 0;
    std::size_t cnt = (nmemb / 16 >= 4) ? (nmemb / 16 - 4) : 0;

    while (cnt > 0) {
        if (static_cast<std::size_t>(pta - ptl) - m <= 48) {
            --cnt;
            for (std::size_t i = 0; i < 16; ++i) {
                bool val = std::invoke(comp, std::invoke(proj, *pta), piv_key);
                if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                    ptl[m] = *pta;
                    ptr[m] = *pta++;
                } else {
                    if (val) {
                        ptl[m] = std::ranges::iter_move(pta++);
                    } else {
                        ptr[m] = std::ranges::iter_move(pta++);
                    }
                }
                m += val;
                --ptr;
            }
        }
        if (cnt == 0) {
            break;
        }
        if (static_cast<std::size_t>(pta - ptl) - m >= 16) {
            --cnt;
            for (std::size_t i = 0; i < 16; ++i) {
                bool val = std::invoke(comp, std::invoke(proj, *tpa), piv_key);
                if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                    ptl[m] = *tpa;
                    ptr[m] = *tpa--;
                } else {
                    if (val) {
                        ptl[m] = std::ranges::iter_move(tpa--);
                    } else {
                        ptr[m] = std::ranges::iter_move(tpa--);
                    }
                }
                m += val;
                --ptr;
            }
        }
    }

    std::size_t rem = nmemb % 16;
    if (static_cast<std::size_t>(pta - ptl) - m <= 48) {
        while (rem-- > 0) {
            bool val = std::invoke(comp, std::invoke(proj, *pta), piv_key);
            if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                ptl[m] = *pta;
                ptr[m] = *pta++;
            } else {
                if (val) {
                    ptl[m] = std::ranges::iter_move(pta++);
                } else {
                    ptr[m] = std::ranges::iter_move(pta++);
                }
            }
            m += val;
            --ptr;
        }
    } else {
        while (rem-- > 0) {
            bool val = std::invoke(comp, std::invoke(proj, *tpa), piv_key);
            if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
                ptl[m] = *tpa;
                ptr[m] = *tpa--;
            } else {
                if (val) {
                    ptl[m] = std::ranges::iter_move(tpa--);
                } else {
                    ptr[m] = std::ranges::iter_move(tpa--);
                }
            }
            m += val;
            --ptr;
        }
    }

    for (std::size_t i = 0; i < 64; ++i) {
        bool val = std::invoke(comp, std::invoke(proj, swap[i]), piv_key);
        if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= 16) {
            ptl[m] = swap[i];
            ptr[m] = swap[i];
        } else {
            if (val) {
                ptl[m] = std::move(swap[i]);
            } else {
                ptr[m] = std::move(swap[i]);
            }
        }
        m += val;
        --ptr;
    }

    guard.dismiss();
    return m;
}

/**
 * @brief Pivot selection supporting median-of-9 and median-of-sqrt(N).
 */
template <std::random_access_iterator I, typename SwapBuf, typename Comp, typename Proj>
I crum_select_pivot(I array, SwapBuf& swap, std::size_t nmemb, bool& generic, const Comp& comp,
                    const Proj& proj) {
    if (nmemb <= 2048) {
        return median_of_nine(array, nmemb, comp, proj);
    }

    // Median of sqrt(N) clamped to swap capacity
    std::size_t sqrt_n = 32;
    while (sqrt_n * sqrt_n < nmemb && sqrt_n * 2 <= swap.size()) {
        sqrt_n *= 2;
    }

    std::size_t div = nmemb / sqrt_n;
    I pta = array + nmemb - 1 - (div / 2);
    I piv = array;

    for (std::size_t cnt = 0; cnt < sqrt_n; ++cnt) {
        std::ranges::iter_swap(piv + cnt, pta);
        pta -= div;
    }

    std::size_t half = sqrt_n / 2;
    tail_swap(piv, swap, half, comp, proj);
    tail_swap(piv + half, swap, half, comp, proj);

    generic =
        !std::invoke(comp, std::invoke(proj, *(piv)), std::invoke(proj, *(piv + sqrt_n - 1))) &&
        !std::invoke(comp, std::invoke(proj, *(piv + sqrt_n - 1)), std::invoke(proj, *(piv)));

    return binary_median(piv, piv + half, half, comp, proj);
}

/**
 * @brief Recursive fulcrum partitioning orchestrator.
 */
template <std::random_access_iterator I, typename SwapBuf, typename Comp, typename Proj>
void fulcrum_partition(I array, SwapBuf& swap, std::size_t nmemb, const Comp& comp,
                       const Proj& proj) {
    while (nmemb > crum_out_threshold) {
        bool generic = false;
        I ptp = crum_select_pivot(array, swap, nmemb, generic, comp, proj);

        if (generic) {
            tail_swap(array, swap, nmemb, comp, proj);
            return;
        }

        auto piv = *ptp;
        std::ranges::iter_swap(ptp, array + nmemb - 1);

        std::size_t a_size = fulcrum_default_partition(array, nmemb - 1, piv, comp, proj);
        std::size_t s_size = (nmemb - 1) - a_size;

        std::ranges::iter_swap(array + a_size, array + nmemb - 1);

        if (a_size <= s_size / 32 || s_size <= crum_out_threshold) {
            tail_swap(array + a_size + 1, swap, s_size, comp, proj);
        } else {
            fulcrum_partition(array + a_size + 1, swap, s_size, comp, proj);
        }

        if (s_size <= a_size / 32 || a_size <= crum_out_threshold) {
            if (a_size <= crum_out_threshold) {
                tail_swap(array, swap, a_size, comp, proj);
            } else {
                a_size = fulcrum_reverse_partition(array, a_size, piv, comp, proj);
                if (a_size <= crum_out_threshold) {
                    tail_swap(array, swap, a_size, comp, proj);
                } else {
                    fulcrum_partition(array, swap, a_size, comp, proj);
                }
            }
            return;
        }

        nmemb = a_size;
    }

    if (nmemb > 1) {
        tail_swap(array, swap, nmemb, comp, proj);
    }
}

} // namespace detail

/**
 * @brief Sorts elements in the range [first, last) in-place using Crumsort.
 *
 * Crumsort is an unstable, branchless in-place hybrid quicksort utilizing fulcrum
 * partitioning with 64-element block evacuation, median-of-sqrt(N) pivot selection,
 * and Quadsort small-sort routines.
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
void crumsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto end_it = std::ranges::next(first, last);
    auto count = static_cast<std::size_t>(end_it - first);

    if (count <= 1) {
        return;
    }

    if (count <= detail::crum_out_threshold) {
        std::array<std::iter_value_t<I>, detail::crum_out_threshold> stack_swap;
        detail::tail_swap(first, stack_swap, count, comp, proj);
        return;
    }

    auto presorted = detail::analyze_presortedness(first, count, comp, proj);
    if (presorted == detail::PresortedResult::AlreadySorted) {
        return;
    }
    if (presorted == detail::PresortedResult::ReverseSorted) {
        std::ranges::reverse(first, end_it);
        return;
    }

    std::array<std::iter_value_t<I>, detail::crum_aux_capacity> stack_swap;
    detail::fulcrum_partition(first, stack_swap, count, comp, proj);
}

/**
 * @brief Sorts a C++20 random access range using Crumsort.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
void crumsort(R&& r, Comp comp = {}, Proj proj = {}) {
    crumsort(std::ranges::begin(r), std::ranges::end(r), std::move(comp), std::move(proj));
}

/**
 * @brief Sorts a contiguous span of elements using Crumsort.
 */
template <typename T, typename Compare = std::ranges::less>
void crumsort(std::span<T> data, Compare comp = {}) {
    crumsort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @struct CrumSort
 * @brief Zero-overhead adapter struct for Crumsort.
 */
struct CrumSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "crumsort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    template <typename T> void sort(std::span<T> data) const {
        crumsort(data);
    }

    template <typename T, typename Compare> void sort(std::span<T> data, Compare comp) const {
        crumsort(data, std::move(comp));
    }
};

using Crumsort = CrumSort;

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "crumsort", ::algoat::sorting::CrumSort)
