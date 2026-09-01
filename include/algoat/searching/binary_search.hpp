/**
 * @file binary_search.hpp
 * @brief Binary Search algorithm implementation.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

namespace algoat::searching {

/**
 * @struct BinarySearch
 * @brief Logarithmic search on sorted spans using midpoint bisection.
 *
 * @par Characteristics:
 * - <b>Preconditions:</b> Range must be sorted in ascending order.
 *
 * @par Time Complexity:
 * - Best Case: @c O(1) (target is at midpoint)
 * - Average Case: @c O(log N)
 * - Worst Case: @c O(log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(1) auxiliary space
 */
struct BinarySearch {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "binarysearch"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "binarysearch";
    }

    /**
     * @brief Searches for target using binary search with overflow-safe midpoint calculation.
     * @tparam T Element type supporting <tt>operator==</tt> and <tt>operator<</tt>.
     *
     * @param data Sorted span of elements.
     *
     * @param target Value to locate.
     * @return Index of a matching element if present, or @c std::nullopt.
     */
    template <typename T>
    std::optional<std::size_t> search(std::span<T> data, const T& target) const {
        return search(std::span<const T>{data.data(), data.size()}, target);
    }

    template <typename T>
    std::optional<std::size_t> search(std::span<const T> data, const T& target) const {
        if (data.empty()) {
            return std::nullopt;
        }

        const T* base = data.data();
        std::size_t range_length = data.size();

        while (range_length > 1) {
            std::size_t half = range_length / 2;
            base = (base[half] < target) ? (base + half) : base;
            range_length -= half;
        }

        std::size_t index = static_cast<std::size_t>(base - data.data());
        if (*base == target) {
            return index;
        }
        if (index + 1 < data.size() && *(base + 1) == target) {
            return index + 1;
        }

        return std::nullopt;
    }

    /**
     * @brief Searches for target across arbitrary random-access indexed sequences branchlessly.
     * @tparam RandomAccessSeq Random access sequence supporting operator[](std::size_t).
     * @tparam T Value type comparable with sequence elements.
     * @param seq The sequence to search.
     * @param size Number of elements in seq.
     * @param target Target value.
     * @return Index of matching element if present, or std::nullopt.
     */
    template <typename RandomAccessSeq, typename T>
    std::optional<std::size_t> search_indexed(const RandomAccessSeq& seq, std::size_t size,
                                              const T& target) const {
        if (size == 0) {
            return std::nullopt;
        }

        std::size_t base = 0;
        std::size_t range_length = size;

        while (range_length > 1) {
            std::size_t half = range_length / 2;
            base = (seq[base + half] < target) ? (base + half) : base;
            range_length -= half;
        }

        if (seq[base] == target) {
            return base;
        }
        if (base + 1 < size && seq[base + 1] == target) {
            return base + 1;
        }

        return std::nullopt;
    }

    /**
     * @brief Indicates whether this search algorithm requires sorted input.
     * @return @c true
     */
    [[nodiscard]] constexpr bool requires_sorted() const noexcept {
        return true;
    }
};

} // namespace algoat::searching
