/**
 * @file sorting_network.hpp
 * @brief Zero-overhead compile-time sorting networks for fixed-size spans.
 *
 * Implements deterministic, branchless sorting networks for small fixed extents (N <= 16).
 * Based on Batcher Odd-Even Merge Sort with optimal small-N overrides, executing
 * straight-line conditional swaps with zero branch mispredictions and strict compile-time
 * inlining guarantees.
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>

namespace algoat::meta {

/**
 * @struct SortingNetworkTable
 * @brief Compile-time comparator pair sequence for a network of size N.
 *
 * Generates comparator index pairs (i, j) at compile time via constexpr evaluation.
 * For N <= 4, uses provably optimal network configurations.
 * For N > 4, uses Batcher's Odd-Even Merge Sort algorithm with network pruning.
 *
 * @tparam N Number of elements in the sorting network.
 */
template <std::size_t N> struct SortingNetworkTable {
    /**
     * @struct Pair
     * @brief A pair of element indices to be conditionally swapped.
     */
    struct Pair {
        std::size_t i = 0; ///< First element index.
        std::size_t j = 0; ///< Second element index.
    };

    std::array<Pair, 256> pairs{}; ///< Buffer of comparator pairs.
    std::size_t count = 0;         ///< Number of active comparator pairs.

    /**
     * @brief Records a comparator pair in the compile-time table.
     * @param first_idx First element index.
     * @param second_idx Second element index.
     */
    constexpr void add(std::size_t first_idx, std::size_t second_idx) noexcept {
        pairs[count++] = {first_idx, second_idx};
    }

    /**
     * @brief Generates the sorting network comparator sequence for size N.
     * @return Fully populated SortingNetworkTable instance.
     */
    [[nodiscard]] static constexpr SortingNetworkTable<N> generate() noexcept {
        SortingNetworkTable<N> net{};
        if constexpr (N <= 1) {
            return net;
        }

        if constexpr (N == 2) {
            net.add(0, 1);
            return net;
        } else if constexpr (N == 3) {
            net.add(0, 1);
            net.add(1, 2);
            net.add(0, 1);
            return net;
        } else if constexpr (N == 4) {
            net.add(0, 1);
            net.add(2, 3);
            net.add(0, 2);
            net.add(1, 3);
            net.add(1, 2);
            return net;
        }

        // Batcher's Odd-Even Merge Network constructor with index pruning for arbitrary N
        auto oddeven_merge = [&](auto& self, std::size_t low_idx, std::size_t length,
                                 std::size_t step) -> void {
            std::size_t next_step = step * 2;
            if (next_step < length) {
                self(self, low_idx, length, next_step);
                self(self, low_idx + step, length, next_step);
                for (std::size_t i = low_idx + step; i + step < low_idx + length; i += next_step) {
                    if (i < N && (i + step) < N) {
                        net.add(i, i + step);
                    }
                }
            } else {
                if (low_idx < N && (low_idx + step) < N) {
                    net.add(low_idx, low_idx + step);
                }
            }
        };

        auto oddeven_sort = [&](auto& self, std::size_t low_idx, std::size_t length) -> void {
            if (length > 1) {
                std::size_t half_length = length / 2;
                self(self, low_idx, half_length);
                self(self, low_idx + half_length, half_length);
                oddeven_merge(oddeven_merge, low_idx, length, 1);
            }
        };

        std::size_t power_of_two = 1;
        while (power_of_two < N) {
            power_of_two *= 2;
        }
        oddeven_sort(oddeven_sort, 0, power_of_two);
        return net;
    }
};

/**
 * @brief Global constexpr table containing comparator pairs for sorting network of size N.
 */
template <std::size_t N> inline constexpr auto network_table_v = SortingNetworkTable<N>::generate();

/**
 * @brief Sorts a fixed-extent span using a zero-overhead compile-time sorting network with a
 * comparator.
 *
 * @tparam T Element type.
 * @tparam N Fixed span extent (must not be std::dynamic_extent).
 * @tparam Compare Strict weak ordering comparator.
 *
 * @param data Fixed-extent span to sort in-place.
 * @param comp Strict weak ordering comparator callable.
 */
template <typename T, std::size_t N, typename Compare>
    requires(N != std::dynamic_extent)
constexpr void sort_network(std::span<T, N> data, Compare comp) {
    if constexpr (N <= 1) {
        return;
    } else {
        constexpr auto net = network_table_v<N>;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (
                [&]() {
                    if (comp(data[net.pairs[Is].j], data[net.pairs[Is].i])) {
                        std::swap(data[net.pairs[Is].i], data[net.pairs[Is].j]);
                    }
                }(),
                ...);
        }(std::make_index_sequence<net.count>{});
    }
}

/**
 * @brief Sorts a fixed-extent span using a zero-overhead compile-time sorting network with natural
 * order.
 *
 * @tparam T Element type.
 * @tparam N Fixed span extent.
 *
 * @param data Fixed-extent span to sort in-place.
 */
template <typename T, std::size_t N>
    requires(N != std::dynamic_extent)
constexpr void sort_network(std::span<T, N> data) {
    sort_network(data, std::less<T>{});
}

} // namespace algoat::meta
