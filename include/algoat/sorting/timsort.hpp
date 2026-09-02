/**
 * @file timsort.hpp
 * @brief High-performance adaptive TimSort hybrid sorting algorithm.
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::sorting {

/**
 * @struct TimSort
 * @brief Adaptive stable hybrid sorting algorithm derived from Merge Sort and Insertion Sort.
 *
 * Implements natural run detection (linear time monotonic identification and in-place reversal
 * of strictly descending sequences), dynamic min-run calculation, stack merge invariants
 * verified against Auger et al. (2018), and galloping mode optimization for high-disparity runs.
 *
 * @par Characteristics:
 * - <b>Category:</b> Hybrid (Insertion Sort + Merge Sort), Adaptive.
 * - <b>Stability:</b> Stable.
 *
 * @par Time Complexity:
 * - Best Case: @c O(N) (pre-sorted or strictly reverse-sorted data)
 * - Average Case: @c O(N log N)
 * - Worst Case: @c O(N log N)
 *
 * @par Space Complexity:
 * - Auxiliary Space: @c O(N) auxiliary buffer space (at most N/2 elements)
 */
struct TimSort {
    /**
     * @brief Returns the unique identifier for this algorithm.
     * @return "timsort"
     */
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "timsort";
    }

    /**
     * @brief Preferred minimum size threshold.
     * @return 0
     */
    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0;
    }

    /**
     * @brief Computes the minimum run length for an array of size n.
     * Keeps min_run between 32 and 64 such that n / min_run is close to a power of 2.
     *
     * @param n Size of data to sort.
     * @return Minimum run size.
     */
    [[nodiscard]] static constexpr std::size_t compute_min_run(std::size_t n) noexcept {
        std::size_t r = 0;
        while (n >= 64) {
            r |= (n & 1);
            n >>= 1;
        }
        return n + r;
    }

    /**
     * @brief Sorts the span in-place using TimSort.
     * @tparam T Type satisfying @c std::totally_ordered.
     *
     * @param data Contiguous span of elements to sort.
     */
    template <std::totally_ordered T> void sort(std::span<T> data) const {
        if (data.size() <= 1) {
            return;
        }

        const std::size_t n = data.size();
        const std::size_t min_run = compute_min_run(n);

        std::vector<Run> pending;
        std::vector<T> temp_buffer;
        temp_buffer.reserve(n / 2);

        std::size_t i = 0;
        while (i < n) {
            std::size_t run_len = count_run_and_make_ascending(data.subspan(i));
            if (run_len < min_run) {
                std::size_t force = std::min(min_run, n - i);
                binary_insertion_sort(data.subspan(i, force), run_len);
                run_len = force;
            }

            pending.push_back(Run{i, run_len});
            merge_collapse(data, pending, temp_buffer);
            i += run_len;
        }

        merge_force_collapse(data, pending, temp_buffer);
    }

private:
    struct Run {
        std::size_t base;
        std::size_t len;
    };

    /**
     * @brief Identifies a natural run starting at index 0 of the span.
     * Strictly descending runs are reversed in-place to maintain stability.
     * Non-descending runs are left intact.
     *
     * @param data Subspan to examine.
     * @return Length of the natural run.
     */
    template <typename T> static std::size_t count_run_and_make_ascending(std::span<T> data) {
        if (data.size() <= 1) {
            return data.size();
        }

        std::size_t run_len = 2;
        if (data[1] < data[0]) {
            // Strictly descending: reverse in-place to preserve stability
            while (run_len < data.size() && data[run_len] < data[run_len - 1]) {
                ++run_len;
            }
            std::reverse(data.begin(), data.begin() + run_len);
        } else {
            // Non-descending
            while (run_len < data.size() && !(data[run_len] < data[run_len - 1])) {
                ++run_len;
            }
        }
        return run_len;
    }

    /**
     * @brief Boosts small runs using binary insertion sort.
     * Elements in [0, start) are assumed already sorted.
     *
     * @param data Subspan to sort.
     * @param start Number of elements already sorted.
     */
    template <typename T> static void binary_insertion_sort(std::span<T> data, std::size_t start) {
        const std::size_t n = data.size();
        for (std::size_t i = start; i < n; ++i) {
            T pivot = std::move(data[i]);
            // upper_bound guarantees stability by inserting after equal keys
            auto it = std::upper_bound(data.begin(), data.begin() + i, pivot);
            std::move_backward(it, data.begin() + i, data.begin() + i + 1);
            *it = std::move(pivot);
        }
    }

    /**
     * @brief Unified exponential search (galloping) using compile-time flags.
     *
     * @tparam IsUpperBound If true, behaves as upper_bound (gallop_right). If false, lower_bound
     * (gallop_left).
     * @tparam IsBackward If true, searches backward from the end of the span.
     */
    template <bool IsUpperBound, bool IsBackward, typename Key, typename Elem>
    static std::size_t gallop(const Key& key, std::span<Elem> span) {
        if (span.empty()) {
            return 0;
        }

        auto comp_front = [&]() {
            if constexpr (IsUpperBound) {
                return key < span.front();
            } else {
                return !(span.front() < key);
            }
        };
        auto comp_back = [&]() {
            if constexpr (IsUpperBound) {
                return !(key < span.back());
            } else {
                return span.back() < key;
            }
        };

        if (comp_front()) {
            return 0;
        }
        if (comp_back()) {
            return span.size();
        }

        const std::size_t n = span.size();
        std::size_t offset = 1;
        std::size_t low = 0;
        std::size_t high = n;

        if constexpr (!IsBackward) {
            auto comp = [&]() {
                if constexpr (IsUpperBound) {
                    return !(key < span[offset]);
                } else {
                    return span[offset] < key;
                }
            };
            while (offset < n && comp()) {
                low = offset;
                offset = (offset << 1) + 1;
            }
            high = std::min(offset, n);
        } else {
            auto comp = [&]() {
                if constexpr (IsUpperBound) {
                    return key < span[n - 1 - offset];
                } else {
                    return !(span[n - 1 - offset] < key);
                }
            };
            while (offset < n && comp()) {
                high = n - offset;
                offset = (offset << 1) + 1;
            }
            low = (offset >= n) ? 0 : (n - offset);
        }

        if constexpr (IsUpperBound) {
            auto it = std::upper_bound(span.begin() + low, span.begin() + high, key);
            return static_cast<std::size_t>(it - span.begin());
        } else {
            auto it = std::lower_bound(span.begin() + low, span.begin() + high, key);
            return static_cast<std::size_t>(it - span.begin());
        }
    }

    /**
     * @brief Merges run A and run B where lenA <= lenB.
     */
    template <typename T>
    static void merge_lo(std::span<T> data, std::size_t baseA, std::size_t lenA, std::size_t baseB,
                         std::size_t lenB, std::vector<T>& temp_buffer) {
        temp_buffer.clear();
        temp_buffer.reserve(lenA);
        for (std::size_t i = 0; i < lenA; ++i) {
            temp_buffer.push_back(std::move(data[baseA + i]));
        }

        std::size_t cursorA = 0;
        std::size_t cursorB = baseB;
        std::size_t dest = baseA;
        const std::size_t endB = baseB + lenB;

        std::size_t min_gallop = 7;
        if constexpr (std::is_fundamental_v<T>) {
            min_gallop = 16;
        }
        if (lenB >= 8 * lenA || lenA >= 8 * lenB) {
            min_gallop = std::min(min_gallop, std::size_t{2});
        }

        while (cursorA < lenA && cursorB < endB) {
            std::size_t countA = 0;
            std::size_t countB = 0;

            // Linear search phase
            while (cursorA < lenA && cursorB < endB) {
                if (data[cursorB] < temp_buffer[cursorA]) {
                    data[dest++] = std::move(data[cursorB++]);
                    ++countB;
                    countA = 0;
                    if (countB >= min_gallop) {
                        break;
                    }
                } else {
                    data[dest++] = std::move(temp_buffer[cursorA++]);
                    ++countA;
                    countB = 0;
                    if (countA >= min_gallop) {
                        break;
                    }
                }
            }

            // Galloping search phase
            while (cursorA < lenA && cursorB < endB) {
                std::span<const T> spanA(temp_buffer.data() + cursorA, lenA - cursorA);
                std::size_t kA = gallop<true, false>(data[cursorB], spanA);
                if (kA > 0) {
                    for (std::size_t m = 0; m < kA; ++m) {
                        data[dest++] = std::move(temp_buffer[cursorA++]);
                    }
                }
                data[dest++] = std::move(data[cursorB++]);

                if (cursorA == lenA || cursorB == endB) {
                    break;
                }

                std::span<const T> spanB(data.data() + cursorB, endB - cursorB);
                std::size_t kB = gallop<false, false>(temp_buffer[cursorA], spanB);
                if (kB > 0) {
                    for (std::size_t m = 0; m < kB; ++m) {
                        data[dest++] = std::move(data[cursorB++]);
                    }
                }
                data[dest++] = std::move(temp_buffer[cursorA++]);

                if (cursorA == lenA || cursorB == endB) {
                    break;
                }

                if (kA < min_gallop || kB < min_gallop) {
                    ++min_gallop;
                    break;
                } else if (min_gallop > 1) {
                    --min_gallop;
                }
            }
        }

        while (cursorA < lenA) {
            data[dest++] = std::move(temp_buffer[cursorA++]);
        }
        temp_buffer.clear();
    }

    /**
     * @brief Merges run A and run B where lenA > lenB.
     */
    template <typename T>
    static void merge_hi(std::span<T> data, std::size_t baseA, std::size_t lenA, std::size_t baseB,
                         std::size_t lenB, std::vector<T>& temp_buffer) {
        temp_buffer.clear();
        temp_buffer.reserve(lenB);
        for (std::size_t i = 0; i < lenB; ++i) {
            temp_buffer.push_back(std::move(data[baseB + i]));
        }

        std::ptrdiff_t cursorA = static_cast<std::ptrdiff_t>(baseA + lenA - 1);
        std::ptrdiff_t cursorB = static_cast<std::ptrdiff_t>(lenB - 1);
        std::ptrdiff_t dest = static_cast<std::ptrdiff_t>(baseB + lenB - 1);
        const std::ptrdiff_t minA = static_cast<std::ptrdiff_t>(baseA);

        std::size_t min_gallop = 7;
        if constexpr (std::is_fundamental_v<T>) {
            min_gallop = 16;
        }
        if (lenB >= 8 * lenA || lenA >= 8 * lenB) {
            min_gallop = std::min(min_gallop, std::size_t{2});
        }

        while (cursorA >= minA && cursorB >= 0) {
            std::size_t countA = 0;
            std::size_t countB = 0;

            // Linear search phase
            while (cursorA >= minA && cursorB >= 0) {
                if (temp_buffer[static_cast<std::size_t>(cursorB)] <
                    data[static_cast<std::size_t>(cursorA)]) {
                    data[static_cast<std::size_t>(dest--)] =
                        std::move(data[static_cast<std::size_t>(cursorA--)]);
                    ++countA;
                    countB = 0;
                    if (countA >= min_gallop) {
                        break;
                    }
                } else {
                    data[static_cast<std::size_t>(dest--)] =
                        std::move(temp_buffer[static_cast<std::size_t>(cursorB--)]);
                    ++countB;
                    countA = 0;
                    if (countB >= min_gallop) {
                        break;
                    }
                }
            }

            // Galloping search phase
            while (cursorA >= minA && cursorB >= 0) {
                const std::size_t remA = static_cast<std::size_t>(cursorA - minA + 1);
                std::span<const T> spanA(data.data() + minA, remA);
                std::size_t kA =
                    gallop<true, true>(temp_buffer[static_cast<std::size_t>(cursorB)], spanA);
                std::size_t countMovedA = remA - kA;
                if (countMovedA > 0) {
                    for (std::size_t m = 0; m < countMovedA; ++m) {
                        data[static_cast<std::size_t>(dest--)] =
                            std::move(data[static_cast<std::size_t>(cursorA--)]);
                    }
                }
                data[static_cast<std::size_t>(dest--)] =
                    std::move(temp_buffer[static_cast<std::size_t>(cursorB--)]);

                if (cursorA < minA || cursorB < 0) {
                    break;
                }

                const std::size_t remB = static_cast<std::size_t>(cursorB + 1);
                std::span<const T> spanB(temp_buffer.data(), remB);
                std::size_t kB =
                    gallop<false, true>(data[static_cast<std::size_t>(cursorA)], spanB);
                std::size_t countMovedB = remB - kB;
                if (countMovedB > 0) {
                    for (std::size_t m = 0; m < countMovedB; ++m) {
                        data[static_cast<std::size_t>(dest--)] =
                            std::move(temp_buffer[static_cast<std::size_t>(cursorB--)]);
                    }
                }
                data[static_cast<std::size_t>(dest--)] =
                    std::move(data[static_cast<std::size_t>(cursorA--)]);

                if (cursorA < minA || cursorB < 0) {
                    break;
                }

                if (countMovedA < min_gallop || countMovedB < min_gallop) {
                    ++min_gallop;
                    break;
                } else if (min_gallop > 1) {
                    --min_gallop;
                }
            }
        }

        while (cursorB >= 0) {
            data[static_cast<std::size_t>(dest--)] =
                std::move(temp_buffer[static_cast<std::size_t>(cursorB--)]);
        }
        temp_buffer.clear();
    }

    /**
     * @brief Merges run at pending[n] with run at pending[n+1].
     */
    template <typename T>
    static void merge_at(std::size_t n, std::span<T> data, std::vector<Run>& pending,
                         std::vector<T>& temp_buffer) {
        std::size_t baseA = pending[n].base;
        std::size_t lenA = pending[n].len;
        std::size_t baseB = pending[n + 1].base;
        std::size_t lenB = pending[n + 1].len;

        pending[n].len = lenA + lenB;
        pending.erase(pending.begin() + n + 1);

        // Trim A: elements at start of A that are <= data[baseB] are already in place
        std::size_t kA = gallop<true, false>(data[baseB], data.subspan(baseA, lenA));
        baseA += kA;
        lenA -= kA;
        if (lenA == 0) {
            return;
        }

        // Trim B: elements at end of B that are >= data[baseA + lenA - 1] are already in place
        std::size_t kB = gallop<false, true>(data[baseA + lenA - 1], data.subspan(baseB, lenB));
        lenB = kB;
        if (lenB == 0) {
            return;
        }

        if (lenA <= lenB) {
            merge_lo(data, baseA, lenA, baseB, lenB, temp_buffer);
        } else {
            merge_hi(data, baseA, lenA, baseB, lenB, temp_buffer);
        }
    }

    /**
     * @brief Collapses pending runs according to Auger et al. (2018) stack invariants.
     */
    template <typename T>
    static void merge_collapse(std::span<T> data, std::vector<Run>& pending,
                               std::vector<T>& temp_buffer) {
        while (pending.size() > 1) {
            std::size_t n = pending.size() - 2;
            if ((n > 0 && pending[n - 1].len <= pending[n].len + pending[n + 1].len) ||
                (n > 1 && pending[n - 2].len <= pending[n - 1].len + pending[n].len)) {
                if (pending[n - 1].len < pending[n + 1].len) {
                    --n;
                }
                merge_at(n, data, pending, temp_buffer);
            } else if (pending[n].len <= pending[n + 1].len) {
                merge_at(n, data, pending, temp_buffer);
            } else {
                break;
            }
        }
    }

    /**
     * @brief Merges all remaining runs until a single sorted sequence remains.
     */
    template <typename T>
    static void merge_force_collapse(std::span<T> data, std::vector<Run>& pending,
                                     std::vector<T>& temp_buffer) {
        while (pending.size() > 1) {
            std::size_t n = pending.size() - 2;
            if (n > 0 && pending[n - 1].len < pending[n + 1].len) {
                --n;
            }
            merge_at(n, data, pending, temp_buffer);
        }
    }
};

} // namespace algoat::sorting
