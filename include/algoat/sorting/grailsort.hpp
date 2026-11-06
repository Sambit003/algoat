/**
 * @file grailsort.hpp
 * @brief GrailSort: In-Place Stable Merge Sort with O(1) Auxiliary Memory.
 *
 * Implements the Huang-Langston (1992) block merge sort engineered by Andrey Astrelin
 * (2013) with formalizations and edge-case corrections from The Holy Grail Sort Project
 * (2020-2021).
 *
 * Characteristics:
 * - Time Complexity: Worst-case O(N log N), Average O(N log N), Best-case O(N)
 * - Space Complexity: Strictly O(1) auxiliary memory (zero heap allocations)
 * - Stability: Unconditionally guaranteed
 * - Comparison Bounds: <= 1.15 N log2 N on typical distributions
 */

#pragma once

#include "algoat/core/registry.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace algoat::sorting {

namespace detail::grail {

enum struct Subarray { LEFT, RIGHT };

template <std::random_access_iterator I> struct State {
    std::iter_difference_t<I> curr_block_len{0};
    Subarray curr_block_origin{Subarray::LEFT};
};

template <std::random_access_iterator I>
constexpr void block_swap(I a, I b, std::iter_difference_t<I> len) {
    for (std::iter_difference_t<I> i = 0; i < len; ++i) {
        std::ranges::iter_swap(a + i, b + i);
    }
}

template <std::random_access_iterator I>
constexpr void rotate(I start, std::iter_difference_t<I> left_len,
                      std::iter_difference_t<I> right_len) {
    while (left_len > 0 && right_len > 0) {
        if (left_len <= right_len) {
            block_swap(start, start + left_len, left_len);
            start += left_len;
            right_len -= left_len;
        } else {
            block_swap(start + (left_len - right_len), start + left_len, right_len);
            left_len -= right_len;
        }
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void insert_sort(I start, I end, Comp& comp, Proj& proj) {
    for (I item = start + 1; item < end; ++item) {
        I right = item;
        while (right > start) {
            I left = right - 1;
            if (std::invoke(comp, std::invoke(proj, *right), std::invoke(proj, *left))) {
                std::ranges::iter_swap(left, right);
                right = left;
            } else {
                break;
            }
        }
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj, typename T>
constexpr std::iter_difference_t<I> binary_search_left(I start, std::iter_difference_t<I> length,
                                                       const T& target, Comp& comp, Proj& proj) {
    std::iter_difference_t<I> left = 0;
    std::iter_difference_t<I> right = length;
    while (left < right) {
        auto middle = left + (right - left) / 2;
        if (std::invoke(comp, std::invoke(proj, *(start + middle)), target)) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

template <std::random_access_iterator I, typename Comp, typename Proj, typename T>
constexpr std::iter_difference_t<I> binary_search_right(I start, std::iter_difference_t<I> length,
                                                        const T& target, Comp& comp, Proj& proj) {
    std::iter_difference_t<I> left = 0;
    std::iter_difference_t<I> right = length;
    while (left < right) {
        auto middle = left + (right - left) / 2;
        if (std::invoke(comp, target, std::invoke(proj, *(start + middle)))) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    return right;
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr std::iter_difference_t<I>
collect_keys(I start, I end, std::iter_difference_t<I> ideal_keys, Comp& comp, Proj& proj) {
    std::iter_difference_t<I> keys_found = 1;
    I first_key = start;
    I curr_key = start + 1;

    while (curr_key < end && keys_found < ideal_keys) {
        auto insert_pos =
            binary_search_left(first_key, keys_found, std::invoke(proj, *curr_key), comp, proj);

        bool is_unique =
            (insert_pos == keys_found) || std::invoke(comp, std::invoke(proj, *curr_key),
                                                      std::invoke(proj, *(first_key + insert_pos)));

        if (is_unique) {
            rotate(first_key, keys_found, curr_key - (first_key + keys_found));
            first_key = curr_key - keys_found;

            for (I k = first_key + keys_found; k > first_key + insert_pos; --k) {
                std::ranges::iter_swap(k, k - 1);
            }
            ++keys_found;
        }
        ++curr_key;
    }

    rotate(start, first_key - start, keys_found);
    return keys_found;
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void pairwise_swaps(I start, std::iter_difference_t<I> length, Comp& comp, Proj& proj) {
    std::iter_difference_t<I> index = 1;
    for (; index < length; index += 2) {
        I left = start + index - 1;
        I right = start + index;
        if (std::invoke(comp, std::invoke(proj, *right), std::invoke(proj, *left))) {
            std::ranges::iter_swap(left - 2, right);
            std::ranges::iter_swap(right - 2, left);
        } else {
            std::ranges::iter_swap(left - 2, left);
            std::ranges::iter_swap(right - 2, right);
        }
    }
    I left = start + index - 1;
    if (left < start + length) {
        std::ranges::iter_swap(left - 2, left);
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void merge_forwards(I start, std::iter_difference_t<I> left_len,
                              std::iter_difference_t<I> right_len,
                              std::iter_difference_t<I> buffer_offset, Comp& comp, Proj& proj) {
    I buffer = start - buffer_offset;
    I left = start;
    I middle = start + left_len;
    I right = middle;
    I end = middle + right_len;

    while (right < end) {
        if (left == middle ||
            std::invoke(comp, std::invoke(proj, *right), std::invoke(proj, *left))) {
            std::ranges::iter_swap(buffer++, right++);
        } else {
            std::ranges::iter_swap(buffer++, left++);
        }
    }

    if (buffer != left) {
        block_swap(buffer, left, middle - left);
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void merge_backwards(I start, std::iter_difference_t<I> left_len,
                               std::iter_difference_t<I> right_len,
                               std::iter_difference_t<I> buffer_offset, Comp& comp, Proj& proj) {
    I left = start + left_len - 1;
    I middle = left;
    I right = middle + right_len;
    I buffer = right + buffer_offset;

    while (left_len > 0) {
        if (right == middle ||
            std::invoke(comp, std::invoke(proj, *right), std::invoke(proj, *left))) {
            std::ranges::iter_swap(buffer--, left);
            --left_len;
            if (left_len > 0) {
                --left;
            }
        } else {
            std::ranges::iter_swap(buffer--, right--);
        }
    }

    if (right != buffer) {
        while (right > middle) {
            std::ranges::iter_swap(buffer--, right--);
        }
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void build_in_place(I start, std::iter_difference_t<I> length,
                              std::iter_difference_t<I> current_len,
                              std::iter_difference_t<I> buffer_len, Comp& comp, Proj& proj) {
    for (auto merge_len = current_len; merge_len < buffer_len; merge_len *= 2) {
        auto full_merge = 2 * merge_len;
        I merge_index = start;
        I merge_end = start + length - full_merge;
        auto buffer_offset = merge_len;

        for (; merge_index <= merge_end; merge_index += full_merge) {
            merge_forwards(merge_index, merge_len, merge_len, buffer_offset, comp, proj);
        }

        auto left_over = length - (merge_index - start);
        if (left_over > merge_len) {
            merge_forwards(merge_index, merge_len, left_over - merge_len, buffer_offset, comp,
                           proj);
        } else {
            rotate(merge_index - merge_len, merge_len, left_over);
        }

        start -= merge_len;
    }

    auto full_merge = 2 * buffer_len;
    auto last_block = length & (full_merge - 1);
    I last_offset = start + length - last_block;

    if (last_block <= buffer_len) {
        rotate(last_offset, last_block, buffer_len);
    } else {
        merge_backwards(last_offset, buffer_len, last_block - buffer_len, buffer_len, comp, proj);
    }

    for (I merge_index = last_offset - full_merge; merge_index >= start;
         merge_index -= full_merge) {
        merge_backwards(merge_index, buffer_len, buffer_len, buffer_len, comp, proj);
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void build_blocks(I start, std::iter_difference_t<I> length,
                            std::iter_difference_t<I> buffer_len, Comp& comp, Proj& proj) {
    pairwise_swaps(start, length, comp, proj);
    build_in_place(start - 2, length, 2, buffer_len, comp, proj);
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr std::iter_difference_t<I>
block_select_sort(I first_key, I start, std::iter_difference_t<I> median_key,
                  std::iter_difference_t<I> block_count, std::iter_difference_t<I> block_len,
                  Comp& comp, Proj& proj) {
    for (std::iter_difference_t<I> first_block = 0; first_block < block_count; ++first_block) {
        std::iter_difference_t<I> select_block = first_block;

        for (std::iter_difference_t<I> curr_block = first_block + 1; curr_block < block_count;
             ++curr_block) {
            I curr_head = start + (curr_block * block_len);
            I select_head = start + (select_block * block_len);

            if (std::invoke(comp, std::invoke(proj, *curr_head), std::invoke(proj, *select_head))) {
                select_block = curr_block;
            } else if (!std::invoke(comp, std::invoke(proj, *select_head),
                                    std::invoke(proj, *curr_head))) {
                if (std::invoke(comp, std::invoke(proj, *(first_key + curr_block)),
                                std::invoke(proj, *(first_key + select_block)))) {
                    select_block = curr_block;
                }
            }
        }

        if (select_block != first_block) {
            block_swap(start + (first_block * block_len), start + (select_block * block_len),
                       block_len);
            std::ranges::iter_swap(first_key + first_block, first_key + select_block);

            if (median_key == first_block) {
                median_key = select_block;
            } else if (median_key == select_block) {
                median_key = first_block;
            }
        }
    }
    return median_key;
}

template <std::random_access_iterator I>
constexpr void in_place_buffer_reset(I start, std::iter_difference_t<I> length,
                                     std::iter_difference_t<I> buffer_offset) {
    I buffer = start + length - 1;
    I index = buffer - buffer_offset;
    while (buffer >= start) {
        std::ranges::iter_swap(buffer, index);
        if (buffer == start) {
            break;
        }
        --buffer;
        --index;
    }
}

template <std::random_access_iterator I>
constexpr void in_place_buffer_rewind(I start, I left_block, I buffer) {
    while (left_block >= start) {
        std::ranges::iter_swap(buffer, left_block);
        if (left_block == start) {
            break;
        }
        --buffer;
        --left_block;
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr Subarray get_subarray(I curr_key, I median_key, Comp& comp, Proj& proj) {
    if (std::invoke(comp, std::invoke(proj, *curr_key), std::invoke(proj, *median_key))) {
        return Subarray::LEFT;
    }
    return Subarray::RIGHT;
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr std::iter_difference_t<I>
count_last_merge_blocks(I offset, std::iter_difference_t<I> block_count,
                        std::iter_difference_t<I> block_len, Comp& comp, Proj& proj) {
    std::iter_difference_t<I> blocks_to_merge = 0;
    I last_right_frag = offset + (block_count * block_len);
    I prev_left_block = last_right_frag - block_len;

    while (blocks_to_merge < block_count && std::invoke(comp, std::invoke(proj, *last_right_frag),
                                                        std::invoke(proj, *prev_left_block))) {
        ++blocks_to_merge;
        prev_left_block -= block_len;
    }
    return blocks_to_merge;
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void lazy_merge(I start, std::iter_difference_t<I> left_len,
                          std::iter_difference_t<I> right_len, Comp& comp, Proj& proj) {
    if (left_len == 0 || right_len == 0) {
        return;
    }
    if (!std::invoke(comp, std::invoke(proj, *(start + left_len)),
                     std::invoke(proj, *(start + left_len - 1)))) {
        return;
    }

    if (left_len < right_len) {
        I middle = start + left_len;
        while (left_len != 0) {
            auto merge_len =
                binary_search_left(middle, right_len, std::invoke(proj, *start), comp, proj);

            if (merge_len != 0) {
                rotate(start, left_len, merge_len);
                start += merge_len;
                middle += merge_len;
                right_len -= merge_len;
            }

            if (right_len == 0) {
                break;
            }
            do {
                ++start;
                --left_len;
            } while (left_len != 0 &&
                     !std::invoke(comp, std::invoke(proj, *middle), std::invoke(proj, *start)));
        }
    } else {
        I end = start + left_len + right_len - 1;
        while (right_len != 0) {
            auto merge_len =
                binary_search_right(start, left_len, std::invoke(proj, *end), comp, proj);

            if (merge_len != left_len) {
                rotate(start + merge_len, left_len - merge_len, right_len);
                end -= (left_len - merge_len);
                left_len = merge_len;
            }

            if (left_len == 0) {
                break;
            }
            I middle = start + left_len;
            while (right_len != 0 &&
                   !std::invoke(comp, std::invoke(proj, *end), std::invoke(proj, *(middle - 1)))) {
                --right_len;
                if (right_len != 0) {
                    --end;
                }
            }
        }
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void smart_lazy_merge(I start, std::iter_difference_t<I> left_len, Subarray left_origin,
                                std::iter_difference_t<I> right_len, State<I>* state, Comp& comp,
                                Proj& proj) {
    I middle = start + left_len;

    if (left_origin == Subarray::LEFT) {
        if (std::invoke(comp, std::invoke(proj, *middle), std::invoke(proj, *(middle - 1)))) {
            while (left_len != 0) {
                auto merge_len =
                    binary_search_left(middle, right_len, std::invoke(proj, *start), comp, proj);

                if (merge_len != 0) {
                    rotate(start, left_len, merge_len);
                    start += merge_len;
                    middle += merge_len;
                    right_len -= merge_len;
                }

                if (right_len == 0) {
                    state->curr_block_len = left_len;
                    return;
                }
                do {
                    ++start;
                    --left_len;
                } while (left_len != 0 &&
                         !std::invoke(comp, std::invoke(proj, *middle), std::invoke(proj, *start)));
            }
        }
    } else {
        if (!std::invoke(comp, std::invoke(proj, *(middle - 1)), std::invoke(proj, *middle))) {
            while (left_len != 0) {
                auto merge_len =
                    binary_search_right(middle, right_len, std::invoke(proj, *start), comp, proj);

                if (merge_len != 0) {
                    rotate(start, left_len, merge_len);
                    start += merge_len;
                    middle += merge_len;
                    right_len -= merge_len;
                }

                if (right_len == 0) {
                    state->curr_block_len = left_len;
                    return;
                }
                do {
                    ++start;
                    --left_len;
                } while (left_len != 0 &&
                         std::invoke(comp, std::invoke(proj, *start), std::invoke(proj, *middle)));
            }
        }
    }

    state->curr_block_len = right_len;
    state->curr_block_origin = (left_origin == Subarray::LEFT) ? Subarray::RIGHT : Subarray::LEFT;
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void smart_merge(I start, std::iter_difference_t<I> left_len, Subarray left_origin,
                           std::iter_difference_t<I> right_len,
                           std::iter_difference_t<I> buffer_offset, State<I>* state, Comp& comp,
                           Proj& proj) {
    I buffer = start - buffer_offset;
    I left = start;
    I middle = start + left_len;
    I right = middle;
    I end = middle + right_len;

    if (left_origin == Subarray::LEFT) {
        while (left < middle && right < end) {
            if (!std::invoke(comp, std::invoke(proj, *right), std::invoke(proj, *left))) {
                std::ranges::iter_swap(buffer++, left++);
            } else {
                std::ranges::iter_swap(buffer++, right++);
            }
        }
    } else {
        while (left < middle && right < end) {
            if (std::invoke(comp, std::invoke(proj, *left), std::invoke(proj, *right))) {
                std::ranges::iter_swap(buffer++, left++);
            } else {
                std::ranges::iter_swap(buffer++, right++);
            }
        }
    }

    if (left < middle) {
        state->curr_block_len = middle - left;
        in_place_buffer_rewind(left, middle - 1, end - 1);
    } else {
        state->curr_block_len = end - right;
        state->curr_block_origin =
            (left_origin == Subarray::LEFT) ? Subarray::RIGHT : Subarray::LEFT;
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void
merge_blocks(I first_key, I median_key, I start, std::iter_difference_t<I> block_count,
             std::iter_difference_t<I> block_len, std::iter_difference_t<I> last_merge_blocks,
             std::iter_difference_t<I> last_len, State<I>* state, Comp& comp, Proj& proj) {
    I next_block = start + block_len;
    state->curr_block_len = block_len;
    state->curr_block_origin = get_subarray(first_key, median_key, comp, proj);

    for (std::iter_difference_t<I> key_index = 1; key_index < block_count;
         ++key_index, next_block += block_len) {
        I curr_block = next_block - state->curr_block_len;
        auto next_block_origin = get_subarray(first_key + key_index, median_key, comp, proj);

        if (next_block_origin == state->curr_block_origin) {
            I buffer = curr_block - block_len;
            block_swap(buffer, curr_block, state->curr_block_len);
            state->curr_block_len = block_len;
        } else {
            smart_merge(curr_block, state->curr_block_len, state->curr_block_origin, block_len,
                        block_len, state, comp, proj);
        }
    }

    I curr_block = next_block - state->curr_block_len;
    I buffer = curr_block - block_len;

    if (last_len != 0) {
        if (state->curr_block_origin == Subarray::RIGHT) {
            block_swap(buffer, curr_block, state->curr_block_len);
            curr_block = next_block;
            state->curr_block_len = block_len * last_merge_blocks;
            state->curr_block_origin = Subarray::LEFT;
        } else {
            state->curr_block_len += block_len * last_merge_blocks;
        }
        merge_forwards(curr_block, state->curr_block_len, last_len, block_len, comp, proj);
    } else {
        block_swap(buffer, curr_block, state->curr_block_len);
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void
lazy_merge_blocks(I first_key, I median_key, I start, std::iter_difference_t<I> block_count,
                  std::iter_difference_t<I> block_len, std::iter_difference_t<I> last_merge_blocks,
                  std::iter_difference_t<I> last_len, State<I>* state, Comp& comp, Proj& proj) {
    I next_block = start + block_len;
    state->curr_block_len = block_len;
    state->curr_block_origin = get_subarray(first_key, median_key, comp, proj);

    for (std::iter_difference_t<I> key_index = 1; key_index < block_count;
         ++key_index, next_block += block_len) {
        I curr_block = next_block - state->curr_block_len;
        auto next_block_origin = get_subarray(first_key + key_index, median_key, comp, proj);

        if (next_block_origin == state->curr_block_origin) {
            state->curr_block_len = block_len;
        } else {
            if (block_len != 0 && state->curr_block_len != 0) {
                smart_lazy_merge(curr_block, state->curr_block_len, state->curr_block_origin,
                                 block_len, state, comp, proj);
            }
        }
    }

    I curr_block = next_block - state->curr_block_len;

    if (last_len != 0) {
        if (state->curr_block_origin == Subarray::RIGHT) {
            curr_block = next_block;
            state->curr_block_len = block_len * last_merge_blocks;
            state->curr_block_origin = Subarray::LEFT;
        } else {
            state->curr_block_len += block_len * last_merge_blocks;
        }
        lazy_merge(curr_block, state->curr_block_len, last_len, comp, proj);
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void
combine_in_place(I first_key, I start, std::iter_difference_t<I> length,
                 std::iter_difference_t<I> subarray_len, std::iter_difference_t<I> block_len,
                 std::iter_difference_t<I> merge_count, std::iter_difference_t<I> last_subarrays,
                 bool use_buffer, State<I>* state, Comp& comp, Proj& proj) {
    auto full_merge = 2 * subarray_len;
    auto block_count = full_merge / block_len;

    for (std::iter_difference_t<I> merge_index = 0; merge_index < merge_count; ++merge_index) {
        I offset = start + (merge_index * full_merge);
        insert_sort(first_key, first_key + block_count, comp, proj);

        auto median_key = subarray_len / block_len;
        median_key =
            block_select_sort(first_key, offset, median_key, block_count, block_len, comp, proj);

        if (use_buffer) {
            merge_blocks(first_key, first_key + median_key, offset, block_count, block_len, 0, 0,
                         state, comp, proj);
        } else {
            lazy_merge_blocks(first_key, first_key + median_key, offset, block_count, block_len, 0,
                              0, state, comp, proj);
        }
    }

    if (last_subarrays != 0) {
        I offset = start + (merge_count * full_merge);
        block_count = last_subarrays / block_len;

        insert_sort(first_key, first_key + block_count + 1, comp, proj);

        auto median_key = subarray_len / block_len;
        median_key =
            block_select_sort(first_key, offset, median_key, block_count, block_len, comp, proj);

        auto last_fragment = last_subarrays - (block_count * block_len);
        std::iter_difference_t<I> last_merge_blocks = 0;
        if (last_fragment != 0) {
            last_merge_blocks = count_last_merge_blocks(offset, block_count, block_len, comp, proj);
        }

        auto smart_merges = block_count - last_merge_blocks;
        if (smart_merges == 0) {
            auto left_len = last_merge_blocks * block_len;
            if (use_buffer) {
                merge_forwards(offset, left_len, last_fragment, block_len, comp, proj);
            } else {
                lazy_merge(offset, left_len, last_fragment, comp, proj);
            }
        } else {
            if (use_buffer) {
                merge_blocks(first_key, first_key + median_key, offset, smart_merges, block_len,
                             last_merge_blocks, last_fragment, state, comp, proj);
            } else {
                lazy_merge_blocks(first_key, first_key + median_key, offset, smart_merges,
                                  block_len, last_merge_blocks, last_fragment, state, comp, proj);
            }
        }
    }

    if (use_buffer) {
        in_place_buffer_reset(start, length, block_len);
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void combine_blocks(I first_key, I start, std::iter_difference_t<I> length,
                              std::iter_difference_t<I> subarray_len,
                              std::iter_difference_t<I> block_len, bool use_buffer, State<I>* state,
                              Comp& comp, Proj& proj) {
    auto full_merge = 2 * subarray_len;
    auto merge_count = length / full_merge;
    auto last_subarrays = length - (full_merge * merge_count);

    if (last_subarrays <= subarray_len) {
        length -= last_subarrays;
        last_subarrays = 0;
    }

    combine_in_place(first_key, start, length, subarray_len, block_len, merge_count, last_subarrays,
                     use_buffer, state, comp, proj);
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void lazy_stable_sort(I start, std::iter_difference_t<I> length, Comp& comp, Proj& proj) {
    I end = start + length;
    for (I left = start, right = start + 1; right < end; left += 2, right += 2) {
        if (std::invoke(comp, std::invoke(proj, *right), std::invoke(proj, *left))) {
            std::ranges::iter_swap(left, right);
        }
    }
    for (std::iter_difference_t<I> merge_len = 2; merge_len < length; merge_len *= 2) {
        auto full_merge = 2 * merge_len;
        std::iter_difference_t<I> merge_index = 0;
        std::iter_difference_t<I> merge_end = length - full_merge;

        for (; merge_index <= merge_end; merge_index += full_merge) {
            lazy_merge(start + merge_index, merge_len, merge_len, comp, proj);
        }

        auto left_over = length - merge_index;
        if (left_over > merge_len) {
            lazy_merge(start + merge_index, merge_len, left_over - merge_len, comp, proj);
        }
    }
}

template <std::random_access_iterator I, typename Comp, typename Proj>
constexpr void common_sort(I start, std::iter_difference_t<I> length, Comp& comp, Proj& proj) {
    if (length < 16) {
        insert_sort(start, start + length, comp, proj);
        return;
    }

    std::iter_difference_t<I> block_len = 1;
    while ((block_len * block_len) < length) {
        block_len *= 2;
    }

    auto key_len = ((length - 1) / block_len) + 1;
    auto ideal_keys = key_len + block_len;

    auto keys_found = collect_keys(start, start + length, ideal_keys, comp, proj);

    bool ideal_buffer = true;
    if (keys_found < ideal_keys) {
        if (keys_found < 4) {
            lazy_stable_sort(start, length, comp, proj);
            return;
        }
        key_len = block_len;
        block_len = 0;
        ideal_buffer = false;
        while (key_len > keys_found) {
            key_len /= 2;
        }
    }

    auto buffer_end = block_len + key_len;
    auto subarray_len = ideal_buffer ? block_len : key_len;

    build_blocks(start + buffer_end, length - buffer_end, subarray_len, comp, proj);

    State<I> state;
    while ((length - buffer_end) > (2 * subarray_len)) {
        subarray_len *= 2;
        auto current_block_len = block_len;
        bool scrolling_buffer = ideal_buffer;

        if (!ideal_buffer) {
            auto key_buffer = key_len / 2;
            if (key_buffer >= ((2 * subarray_len) / key_buffer)) {
                current_block_len = key_buffer;
                scrolling_buffer = true;
            } else {
                current_block_len = (2 * subarray_len) / key_len;
            }
        }

        combine_blocks(start, start + buffer_end, length - buffer_end, subarray_len,
                       current_block_len, scrolling_buffer, &state, comp, proj);
    }

    insert_sort(start, start + buffer_end, comp, proj);
    lazy_merge(start, buffer_end, length - buffer_end, comp, proj);
}

} // namespace detail::grail

/**
 * @brief Sorts the given random access range in-place using GrailSort.
 *
 * @tparam I Random access iterator type.
 * @tparam S Sentinel type for iterator I.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable applied to elements before comparison.
 * @param first Beginning of range to sort.
 * @param last Sentinel denoting end of range.
 * @param comp Comparator callable (default: std::ranges::less).
 * @param proj Projection callable (default: std::identity).
 * @return Iterator pointing to the end of the range.
 */
template <std::random_access_iterator I, std::sentinel_for<I> S, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<I, Comp, Proj>
constexpr I grailsort(I first, S last, Comp comp = {}, Proj proj = {}) {
    auto n = std::ranges::distance(first, last);
    if (n > 1) {
        detail::grail::common_sort(first, n, comp, proj);
    }
    return first + n;
}

/**
 * @brief Sorts the given random access range in-place using GrailSort.
 *
 * @tparam R Random access range type.
 * @tparam Comp Strict weak ordering comparator.
 * @tparam Proj Projection callable applied to elements before comparison.
 * @param r Range to sort.
 * @param comp Comparator callable (default: std::ranges::less).
 * @param proj Projection callable (default: std::identity).
 * @return Borrowed iterator pointing to the end of the range.
 */
template <std::ranges::random_access_range R, typename Comp = std::ranges::less,
          typename Proj = std::identity>
    requires std::sortable<std::ranges::iterator_t<R>, Comp, Proj>
constexpr std::ranges::borrowed_iterator_t<R> grailsort(R&& r, Comp comp = {}, Proj proj = {}) {
    return grailsort(std::ranges::begin(r), std::ranges::end(r), std::move(comp), std::move(proj));
}

/**
 * @brief Sorts a contiguous span in-place using GrailSort with a custom comparator.
 *
 * @tparam T Element type.
 * @tparam Compare Strict weak ordering comparator.
 * @param data Contiguous span of elements to sort.
 * @param comp Comparator callable.
 */
template <typename T, typename Compare> constexpr void grailsort(std::span<T> data, Compare comp) {
    grailsort(data.begin(), data.end(), std::move(comp), std::identity{});
}

/**
 * @brief Sorts a contiguous span in-place using GrailSort.
 *
 * @tparam T Element type supporting operator<.
 * @param data Contiguous span of elements to sort.
 */
template <typename T> constexpr void grailsort(std::span<T> data) {
    grailsort(data.begin(), data.end(), std::ranges::less{}, std::identity{});
}

/**
 * @struct GrailSort
 * @brief Zero-overhead adapter struct for GrailSort satisfying SortAlgorithm concept.
 */
struct GrailSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "grailsort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 64;
    }

    template <typename T> constexpr void sort(std::span<T> data) const {
        grailsort(data);
    }

    template <typename T, typename Compare>
    constexpr void sort(std::span<T> data, Compare comp) const {
        grailsort(data, std::move(comp));
    }
};

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "grailsort", ::algoat::sorting::GrailSort)
