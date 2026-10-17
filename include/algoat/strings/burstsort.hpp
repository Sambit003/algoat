#pragma once

#include "algoat/strings/multikey_quicksort.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace algoat::strings {

namespace detail {

inline constexpr std::size_t DEFAULT_BURST_THRESHOLD = 8192;
inline constexpr std::size_t MAX_BURST_DEPTH = 4096;

template <typename StringType> struct TrieNode;

template <typename StringType> struct BurstElement {
    enum class Kind : uint8_t { Bucket, Node };
    Kind kind;
    explicit BurstElement(Kind k) noexcept : kind(k) {}
    virtual ~BurstElement() = default;
};

template <typename StringType> struct BurstBucket final : BurstElement<StringType> {
    std::vector<StringType> items;
    bool burst_disabled{false};

    BurstBucket() noexcept : BurstElement<StringType>(BurstElement<StringType>::Kind::Bucket) {
        items.reserve(64);
    }
};

template <typename StringType> struct TrieNode final : BurstElement<StringType> {
    std::array<std::unique_ptr<BurstElement<StringType>>, 256> children{};
    std::vector<StringType> ended;

    TrieNode() noexcept : BurstElement<StringType>(BurstElement<StringType>::Kind::Node) {}
};

template <typename StringType>
void burst_bucket(std::unique_ptr<BurstElement<StringType>>& element_slot, std::size_t depth,
                  std::size_t threshold) {
    auto* bucket = static_cast<BurstBucket<StringType>*>(element_slot.get());
    if (bucket->items.empty()) {
        return;
    }

    if (depth >= MAX_BURST_DEPTH) {
        bucket->burst_disabled = true;
        return;
    }

    // Check if all items in the bucket have the exact same character/end at next depth
    const int first_char = char_at(bucket->items[0], depth);
    bool all_same = true;
    for (std::size_t i = 1; i < bucket->items.size(); ++i) {
        if (char_at(bucket->items[i], depth) != first_char) {
            all_same = false;
            break;
        }
    }

    if (all_same) {
        if (first_char == -1) {
            bucket->burst_disabled = true;
            return;
        }
        // If all characters match, check if all full strings are identical
        auto sv0 = to_string_view(bucket->items[0]);
        bool all_identical = true;
        for (std::size_t i = 1; i < bucket->items.size(); ++i) {
            if (to_string_view(bucket->items[i]) != sv0) {
                all_identical = false;
                break;
            }
        }
        if (all_identical) {
            bucket->burst_disabled = true;
            return;
        }
    }

    auto new_node = std::make_unique<TrieNode<StringType>>();

    // Move elements from overflowing bucket into the new trie node
    auto items = std::move(bucket->items);
    for (auto& item : items) {
        const int c = char_at(item, depth);
        if (c == -1) {
            new_node->ended.push_back(std::move(item));
        } else {
            const auto byte_val = static_cast<std::size_t>(c);
            if (!new_node->children[byte_val]) {
                new_node->children[byte_val] = std::make_unique<BurstBucket<StringType>>();
            }
            auto* child_bucket =
                static_cast<BurstBucket<StringType>*>(new_node->children[byte_val].get());
            child_bucket->items.push_back(std::move(item));
        }
    }

    // Replace the slot with the new trie node
    element_slot = std::move(new_node);
}

template <typename StringType>
void insert_into_trie(std::unique_ptr<BurstElement<StringType>>& root, StringType item,
                      std::size_t threshold) {
    if (!root) {
        root = std::make_unique<BurstBucket<StringType>>();
    }

    std::unique_ptr<BurstElement<StringType>>* slot = &root;
    std::size_t depth = 0;

    while (true) {
        if ((*slot)->kind == BurstElement<StringType>::Kind::Bucket) {
            auto* bucket = static_cast<BurstBucket<StringType>*>(slot->get());
            bucket->items.push_back(std::move(item));

            if (bucket->items.size() > threshold && !bucket->burst_disabled) {
                burst_bucket(*slot, depth, threshold);
            }
            return;
        }

        auto* node = static_cast<TrieNode<StringType>*>(slot->get());
        const int c = char_at(item, depth);
        if (c == -1) {
            node->ended.push_back(std::move(item));
            return;
        }

        const auto byte_val = static_cast<std::size_t>(c);
        if (!node->children[byte_val]) {
            node->children[byte_val] = std::make_unique<BurstBucket<StringType>>();
        }
        slot = &node->children[byte_val];
        ++depth;
    }
}

template <typename StringType>
void extract_sorted(BurstElement<StringType>* element, std::span<StringType> dest,
                    std::size_t& dest_idx, std::size_t depth) {
    if (!element) {
        return;
    }

    if (element->kind == BurstElement<StringType>::Kind::Bucket) {
        auto* bucket = static_cast<BurstBucket<StringType>*>(element);
        if (bucket->items.size() > 1) {
            multikey_quicksort_impl(std::span<StringType>{bucket->items}, depth);
        }
        for (auto& item : bucket->items) {
            dest[dest_idx++] = std::move(item);
        }
    } else {
        auto* node = static_cast<TrieNode<StringType>*>(element);

        // Strings that terminate at this prefix come first lexicographically
        for (auto& item : node->ended) {
            dest[dest_idx++] = std::move(item);
        }

        // Traverse 256 children in ascending byte order
        for (std::size_t b = 0; b < 256; ++b) {
            if (node->children[b]) {
                extract_sorted(node->children[b].get(), dest, dest_idx, depth + 1);
            }
        }
    }
}

} // namespace detail

/**
 * @brief Sorts a span of strings using Burstsort (Sinha & Zobel 2004).
 *
 * Employs a dynamic cache-conscious access trie with contiguous bucket containers
 * bursting at threshold L = 8,192 into 256-ary subtries. Achieves superior L1/L2 cache
 * locality on large string collections.
 *
 * @tparam StringType String element type (default: std::string_view).
 * @param strings Contiguous span of string-like elements to sort in-place.
 */
template <typename StringType = std::string_view> void burstsort(std::span<StringType> strings) {
    const std::size_t n = strings.size();
    if (n <= 1) {
        return;
    }

    // For very small spans, bypass trie construction overhead and use Multi-Key Quicksort directly
    if (n <= 64) {
        multikey_quicksort(strings);
        return;
    }

    std::unique_ptr<detail::BurstElement<StringType>> root;

    for (std::size_t i = 0; i < n; ++i) {
        detail::insert_into_trie(root, std::move(strings[i]), detail::DEFAULT_BURST_THRESHOLD);
    }

    std::size_t dest_idx = 0;
    detail::extract_sorted(root.get(), strings, dest_idx, 0);
}

/**
 * @brief Sorts a contiguous random-access range of strings in-place using Burstsort.
 *
 * @tparam R Random-access range type with string elements.
 * @param range Range to sort in-place.
 */
template <std::ranges::random_access_range R>
    requires(!detail::is_span_v<std::remove_cvref_t<R>>)
void burstsort(R&& range) {
    if (std::ranges::size(range) <= 1) {
        return;
    }
    using ValueType = std::ranges::range_value_t<R>;
    if constexpr (std::ranges::contiguous_range<R>) {
        burstsort(std::span<ValueType>(std::ranges::data(range), std::ranges::size(range)));
    } else {
        std::vector<ValueType> copy(std::ranges::begin(range), std::ranges::end(range));
        burstsort(std::span<ValueType>(copy));
        std::ranges::copy(copy, std::ranges::begin(range));
    }
}

} // namespace algoat::strings
