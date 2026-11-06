/**
 * @file ska_sort.hpp
 * @brief Ska Sort: High-performance in-place MSD Radix Sort with O(1) auxiliary heap memory.
 *
 * Implements Malte Skarupke's (2016) Ska Sort algorithm based on American Flag Sort
 * (McIlroy, Bostic, & McIlroy 1993). Combines in-place byte permutation cycles,
 * 4-way loop unrolling for high instruction-level parallelism (ILP), adaptive
 * hybrid fallback to Pattern-Defeating Quicksort (pdqsort) for sub-partitions N <= 128,
 * and full C++20 Ranges projection support.
 */

#pragma once

#include "algoat/core/registry.hpp"
#include "algoat/sorting/pdqsort.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace algoat::sorting {

namespace detail {

inline constexpr std::size_t ska_pdqsort_threshold = 128;
inline constexpr std::size_t ska_american_flag_threshold = 1024;

/**
 * @struct PartitionInfo
 * @brief Bookkeeping node for 256 byte partitions.
 */
struct PartitionInfo {
    union {
        std::size_t count{0};
        std::size_t offset;
    };
    std::size_t next_offset{0};
};

/**
 * @brief Unrolls a loop 4 times for high instruction-level parallelism.
 */
template <typename It, typename Func>
constexpr void unroll_loop_four_times(It begin, std::size_t count, Func&& func) {
    std::size_t unrolled = count / 4;
    for (std::size_t i = 0; i < unrolled; ++i) {
        func(begin);
        ++begin;
        func(begin);
        ++begin;
        func(begin);
        ++begin;
        func(begin);
        ++begin;
    }
    std::size_t rem = count % 4;
    for (std::size_t i = 0; i < rem; ++i) {
        func(begin);
        ++begin;
    }
}

/**
 * @brief In-place forward partitioning without standard library unrolling overhead.
 */
template <typename It, typename Pred>
constexpr It inplace_forward_partition(It begin, It end, Pred&& pred) {
    while (begin != end && pred(*begin)) {
        ++begin;
    }
    if (begin == end) {
        return begin;
    }
    for (It it = std::next(begin); it != end; ++it) {
        if (pred(*it)) {
            std::iter_swap(begin, it);
            ++begin;
        }
    }
    return begin;
}

// -----------------------------------------------------------------------------
// Radix Key Conversion Traits
// -----------------------------------------------------------------------------

template <typename T> struct is_tuple_like : std::false_type {};

template <typename T1, typename T2> struct is_tuple_like<std::pair<T1, T2>> : std::true_type {};

template <typename... Ts> struct is_tuple_like<std::tuple<Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool is_tuple_like_v = is_tuple_like<std::remove_cvref_t<T>>::value;

template <typename T>
struct is_radix_sortable_scalar
    : std::bool_constant<std::is_integral_v<T> || std::is_floating_point_v<T> ||
                         std::is_enum_v<T> || std::is_pointer_v<T>> {};

template <typename T>
struct is_radix_sortable : is_radix_sortable_scalar<std::remove_cvref_t<T>> {};

template <typename T1, typename T2>
struct is_radix_sortable<std::pair<T1, T2>>
    : std::bool_constant<is_radix_sortable<T1>::value && is_radix_sortable<T2>::value> {};

template <typename... Ts>
struct is_radix_sortable<std::tuple<Ts...>>
    : std::bool_constant<(is_radix_sortable<Ts>::value && ...)> {};

template <typename T>
inline constexpr bool is_radix_sortable_v = is_radix_sortable<std::remove_cvref_t<T>>::value;

/**
 * @brief Maps scalar values to unsigned sortable representations.
 */
template <typename K> [[nodiscard]] constexpr auto to_sortable_scalar_key(K key) noexcept {
    using Raw = std::remove_cvref_t<K>;
    if constexpr (std::is_same_v<Raw, bool>) {
        return static_cast<uint8_t>(key ? 1 : 0);
    } else if constexpr (std::is_floating_point_v<Raw>) {
        if constexpr (sizeof(Raw) == 4) {
            uint32_t u = std::bit_cast<uint32_t>(key);
            uint32_t sign_bit = -static_cast<int32_t>(u >> 31);
            return u ^ (sign_bit | 0x80000000u);
        } else if constexpr (sizeof(Raw) == 8) {
            uint64_t u = std::bit_cast<uint64_t>(key);
            uint64_t sign_bit = -static_cast<int64_t>(u >> 63);
            return u ^ (sign_bit | (1ULL << 63));
        } else {
            static_assert(sizeof(Raw) == 4 || sizeof(Raw) == 8,
                          "Unsupported floating-point width for ska_sort");
        }
    } else if constexpr (std::is_integral_v<Raw>) {
        using U = std::make_unsigned_t<Raw>;
        U u = static_cast<U>(key);
        if constexpr (std::is_signed_v<Raw>) {
            constexpr U sign_mask = U(1) << (sizeof(U) * 8 - 1);
            return static_cast<U>(u ^ sign_mask);
        } else {
            return u;
        }
    } else if constexpr (std::is_pointer_v<Raw>) {
        return reinterpret_cast<std::uintptr_t>(key);
    } else if constexpr (std::is_enum_v<Raw>) {
        return to_sortable_scalar_key(static_cast<std::underlying_type_t<Raw>>(key));
    } else {
        return key;
    }
}

template <typename Tuple, std::size_t... Is>
constexpr std::size_t tuple_total_bytes_impl(std::index_sequence<Is...>) noexcept {
    return (sizeof(decltype(to_sortable_scalar_key(std::get<Is>(std::declval<Tuple>())))) + ...);
}

template <typename Tuple> constexpr std::size_t tuple_total_bytes() noexcept {
    return tuple_total_bytes_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <std::size_t I, typename Tuple>
constexpr uint8_t extract_tuple_byte_helper(const Tuple& t, std::size_t byte_idx) noexcept {
    if constexpr (I >= std::tuple_size_v<Tuple>) {
        return 0;
    } else {
        auto key = to_sortable_scalar_key(std::get<I>(t));
        constexpr std::size_t elem_size = sizeof(key);
        if (byte_idx < elem_size) {
            std::size_t shift = (elem_size - 1 - byte_idx) * 8;
            return static_cast<uint8_t>((key >> shift) & 0xFF);
        } else {
            return extract_tuple_byte_helper<I + 1>(t, byte_idx - elem_size);
        }
    }
}

template <typename Key> constexpr std::size_t total_radix_bytes() noexcept {
    using Raw = std::remove_cvref_t<Key>;
    if constexpr (is_tuple_like_v<Raw>) {
        return tuple_total_bytes<Raw>();
    } else {
        return sizeof(decltype(to_sortable_scalar_key(std::declval<Raw>())));
    }
}

template <typename Key>
constexpr uint8_t extract_byte_from_key(const Key& key, std::size_t byte_idx) noexcept {
    using Raw = std::remove_cvref_t<Key>;
    if constexpr (is_tuple_like_v<Raw>) {
        return extract_tuple_byte_helper<0>(key, byte_idx);
    } else {
        auto s_key = to_sortable_scalar_key(key);
        constexpr std::size_t K = sizeof(s_key);
        std::size_t shift = (K - 1 - byte_idx) * 8;
        return static_cast<uint8_t>((s_key >> shift) & 0xFF);
    }
}

template <typename T, typename Proj>
constexpr uint8_t get_radix_byte(const T& elem, Proj& proj, std::size_t byte_idx) noexcept {
    return extract_byte_from_key(std::invoke(proj, elem), byte_idx);
}

template <typename ProjectedKey> constexpr auto make_radix_comp() noexcept {
    return [](const auto& a, const auto& b) {
        if constexpr (is_radix_sortable_scalar<ProjectedKey>::value) {
            return to_sortable_scalar_key(a) < to_sortable_scalar_key(b);
        } else {
            return std::ranges::less{}(a, b);
        }
    };
}

// -----------------------------------------------------------------------------
// In-Place Partitioning Engines
// -----------------------------------------------------------------------------

template <typename T, typename Proj>
void ska_sort_impl(std::span<T> data, Proj& proj, std::size_t byte_idx, std::size_t total_bytes);

template <typename T, typename Proj>
int build_partition_offsets(std::span<T> data, Proj& proj, std::size_t byte_idx,
                            PartitionInfo partitions[256],
                            uint8_t remaining_partitions[256]) noexcept {
    for (const auto& elem : data) {
        ++partitions[get_radix_byte(elem, proj, byte_idx)].count;
    }

    std::size_t total = 0;
    int num_partitions = 0;
    for (int i = 0; i < 256; ++i) {
        std::size_t count = partitions[i].count;
        if (count) {
            partitions[i].offset = total;
            total += count;
            remaining_partitions[num_partitions] = static_cast<uint8_t>(i);
            ++num_partitions;
        }
        partitions[i].next_offset = total;
    }
    return num_partitions;
}

template <typename T, typename Proj>
void dispatch_sub_buckets(std::span<T> data, Proj& proj, std::size_t byte_idx,
                          std::size_t total_bytes, const PartitionInfo partitions[256]) {
    if (byte_idx + 1 >= total_bytes) {
        return;
    }
    using ProjectedKey = std::remove_cvref_t<std::invoke_result_t<Proj&, const T&>>;
    auto radix_comp = make_radix_comp<ProjectedKey>();

    std::size_t prev_end = 0;
    for (int p = 0; p < 256; ++p) {
        std::size_t curr_end = partitions[p].next_offset;
        std::size_t bucket_size = curr_end - prev_end;
        if (bucket_size > 1) {
            if (bucket_size <= ska_pdqsort_threshold) {
                pdqsort(data.begin() + prev_end, data.begin() + curr_end, radix_comp, proj);
            } else {
                ska_sort_impl(data.subspan(prev_end, bucket_size), proj, byte_idx + 1, total_bytes);
            }
        }
        prev_end = curr_end;
    }
}

template <typename T, typename Proj>
void american_flag_sort_pass(std::span<T> data, Proj& proj, std::size_t byte_idx,
                             std::size_t total_bytes) {
    PartitionInfo partitions[256];
    uint8_t remaining_partitions[256];
    int num_partitions =
        build_partition_offsets(data, proj, byte_idx, partitions, remaining_partitions);

    if (num_partitions <= 1) {
        if (byte_idx + 1 < total_bytes) {
            ska_sort_impl(data, proj, byte_idx + 1, total_bytes);
        }
        return;
    }

    uint8_t* current_block_ptr = remaining_partitions;
    PartitionInfo* current_block = partitions + *current_block_ptr;
    uint8_t* last_block = remaining_partitions + num_partitions - 1;
    auto it = data.begin();
    auto block_end = data.begin() + current_block->next_offset;
    auto last_element = data.end() - 1;

    for (;;) {
        PartitionInfo* block = partitions + get_radix_byte(*it, proj, byte_idx);
        if (block == current_block) {
            ++it;
            if (it == last_element) {
                break;
            }
            if (it == block_end) {
                for (;;) {
                    ++current_block_ptr;
                    if (current_block_ptr == last_block) {
                        goto recurse;
                    }
                    current_block = partitions + *current_block_ptr;
                    if (current_block->offset != current_block->next_offset) {
                        break;
                    }
                }
                it = data.begin() + current_block->offset;
                block_end = data.begin() + current_block->next_offset;
            }
        } else {
            std::size_t offset = block->offset++;
            std::iter_swap(it, data.begin() + offset);
        }
    }

recurse:
    dispatch_sub_buckets(data, proj, byte_idx, total_bytes, partitions);
}

template <typename T, typename Proj>
void ska_byte_sort_pass(std::span<T> data, Proj& proj, std::size_t byte_idx,
                        std::size_t total_bytes) {
    PartitionInfo partitions[256];
    uint8_t remaining_partitions[256];
    int num_partitions =
        build_partition_offsets(data, proj, byte_idx, partitions, remaining_partitions);

    if (num_partitions <= 1) {
        if (byte_idx + 1 < total_bytes) {
            ska_sort_impl(data, proj, byte_idx + 1, total_bytes);
        }
        return;
    }

    for (uint8_t *last_remaining = remaining_partitions + num_partitions,
                 *end_partition = remaining_partitions + 1;
         last_remaining > end_partition;) {
        last_remaining =
            inplace_forward_partition(remaining_partitions, last_remaining, [&](uint8_t partition) {
                std::size_t& begin_offset = partitions[partition].offset;
                std::size_t& end_offset = partitions[partition].next_offset;
                if (begin_offset == end_offset) {
                    return false;
                }

                unroll_loop_four_times(data.begin() + begin_offset, end_offset - begin_offset,
                                       [&](auto it) {
                                           uint8_t this_part = get_radix_byte(*it, proj, byte_idx);
                                           std::size_t offset = partitions[this_part].offset++;
                                           std::iter_swap(it, data.begin() + offset);
                                       });
                return begin_offset != end_offset;
            });
    }

    dispatch_sub_buckets(data, proj, byte_idx, total_bytes, partitions);
}

template <typename T, typename Proj>
void ska_sort_impl(std::span<T> data, Proj& proj, std::size_t byte_idx, std::size_t total_bytes) {
    if (data.size() <= ska_pdqsort_threshold) {
        using ProjectedKey = std::remove_cvref_t<std::invoke_result_t<Proj&, const T&>>;
        pdqsort(data.begin(), data.end(), make_radix_comp<ProjectedKey>(), proj);
        return;
    }
    if (data.size() < ska_american_flag_threshold) {
        american_flag_sort_pass(data, proj, byte_idx, total_bytes);
    } else {
        ska_byte_sort_pass(data, proj, byte_idx, total_bytes);
    }
}

} // namespace detail

// -----------------------------------------------------------------------------
// Public API Functions
// -----------------------------------------------------------------------------

/**
 * @brief Sorts a range delimited by iterators using Ska Sort or pdqsort fallback.
 *
 * @tparam I Random access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam Proj Key extractor projection.
 * @param first Beginning of range to sort.
 * @param last Sentinel / end of range.
 * @param proj Projection callable.
 * @return Iterator pointing to the sorted range end.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Proj = std::identity>
    requires std::sortable<I, std::ranges::less, Proj>
constexpr I ska_sort(I first, S last, Proj proj = {}) {
    auto last_it = std::ranges::next(first, last);
    auto count = last_it - first;
    if (count <= 1) {
        return last_it;
    }

    if (std::is_constant_evaluated()) {
        pdqsort(first, last_it, std::ranges::less{}, proj);
        return last_it;
    }

    using ProjectedKey = std::remove_cvref_t<std::invoke_result_t<Proj&, std::iter_reference_t<I>>>;
    if constexpr (!std::contiguous_iterator<I> || !detail::is_radix_sortable_v<ProjectedKey>) {
        pdqsort(first, last_it, std::ranges::less{}, proj);
    } else {
        auto radix_comp = detail::make_radix_comp<ProjectedKey>();
        if (static_cast<std::size_t>(count) <= detail::ska_pdqsort_threshold) {
            pdqsort(first, last_it, radix_comp, proj);
        } else {
            using ValueType = std::iter_value_t<I>;
            std::span<ValueType> data_span(std::to_address(first), static_cast<std::size_t>(count));
            constexpr std::size_t total_bytes = detail::total_radix_bytes<ProjectedKey>();
            detail::ska_sort_impl(data_span, proj, 0, total_bytes);
        }
    }
    return last_it;
}

/**
 * @brief Sorts a contiguous span in-place using Ska Sort with projection.
 *
 * @tparam T Element type.
 * @tparam Proj Key extractor projection.
 * @param data Contiguous span of elements.
 * @param proj Projection callable.
 */
template <typename T, typename Proj> void ska_sort(std::span<T> data, Proj proj) {
    ska_sort(data.begin(), data.end(), std::move(proj));
}

/**
 * @brief Sorts a contiguous span in-place using Ska Sort with identity projection.
 *
 * @tparam T Element type.
 * @param data Contiguous span of elements.
 */
template <typename T> void ska_sort(std::span<T> data) {
    ska_sort(data.begin(), data.end(), std::identity{});
}

/**
 * @brief Sorts a C++20 random access range in-place with optional key projection.
 *
 * @tparam R Random access range type.
 * @tparam Proj Key extractor projection.
 * @param data Range of elements to sort.
 * @param proj Projection callable.
 * @return Borrowed iterator pointing to the end of data (or std::ranges::dangling).
 */
template <std::ranges::random_access_range R, typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, std::ranges::less, Proj>
constexpr std::ranges::borrowed_iterator_t<R> ska_sort(R&& data, Proj proj = {}) {
    ska_sort(std::ranges::begin(data), std::ranges::end(data), std::move(proj));
    return std::ranges::next(std::ranges::begin(data), std::ranges::end(data));
}

/**
 * @struct SkaSort
 * @brief Zero-overhead adapter struct for Ska Sort conforming to the SortAlgorithm concept.
 */
struct SkaSort {
    /**
     * @brief Unique algorithm name identifier.
     * @return "ska_sort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "ska_sort";
    }

    /**
     * @brief Minimum recommended element threshold.
     * @return 128
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 128;
    }

    /**
     * @brief Sorts the span in-place using default projection.
     * @tparam T Element type.
     * @param data Contiguous span of elements to sort.
     */
    template <typename T> void sort(std::span<T> data) const {
        ska_sort(data);
    }
};

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "ska_sort", ::algoat::sorting::SkaSort)
ALGOAT_REGISTER_ALGORITHM("sorting", "skasort", ::algoat::sorting::SkaSort)
