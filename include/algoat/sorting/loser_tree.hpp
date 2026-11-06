/**
 * @file loser_tree.hpp
 * @brief Tournament Loser Tree K-Way Stream Merging Engine.
 *
 * Implements an optimal tournament loser tree data structure and multi-way
 * merge helper function for merging K sorted sequences, following Knuth (TAOCP Vol 3 §5.4.1),
 * Graefe (2006), and Sanders (2000).
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace algoat::sorting {

/**
 * @class LoserTree
 * @brief Tournament Loser Tree for K-way stream merging with O(log K) comparisons.
 *
 * Each internal node stores the loser of a match between competing subtrees, while
 * the overall tournament champion resides at index 0. Replaying the tree upon element
 * replacement requires strictly <= ceil(log2(K)) comparisons, exactly half that of
 * a traditional binary min-heap.
 *
 * @tparam ElementType The type of elements being merged.
 * @tparam Compare Strict weak ordering comparator (defaults to std::ranges::less).
 */
template <typename ElementType, typename Compare = std::ranges::less> class LoserTree {
public:
    using value_type = ElementType;
    using compare_type = Compare;

    /**
     * @brief Constructs a LoserTree for k_streams input streams.
     * @param k_streams Total number of candidate input streams.
     * @param comp Comparison functor.
     */
    explicit LoserTree(std::size_t k_streams, Compare comp = Compare{})
        : k_(k_streams), active_streams_(0), comp_(std::move(comp)), tree_(k_streams, 0), leaves_(),
          exhausted_(k_streams, 0) {
        leaves_.reserve(k_streams);
    }

    /**
     * @brief Initializes or resets stream front values.
     *
     * Constructs the tournament tree in O(K) comparisons using bottom-up divide-and-conquer.
     *
     * @param stream_heads Head elements for each of the streams (size <= k_streams).
     */
    void initialize(std::span<const ElementType> stream_heads) {
        const std::size_t n = std::min(stream_heads.size(), k_);
        leaves_.assign(stream_heads.begin(), stream_heads.begin() + n);

        std::fill(exhausted_.begin(), exhausted_.end(), static_cast<std::uint8_t>(0));
        for (std::size_t i = n; i < k_; ++i) {
            exhausted_[i] = 1;
        }
        active_streams_ = n;

        if (k_ == 0 || active_streams_ == 0) {
            return;
        }
        if (k_ == 1) {
            tree_[0] = 0;
            return;
        }

        // Bottom-up initialization in O(K) comparisons
        std::vector<std::size_t> winner(k_, 0);
        for (std::size_t p = k_ - 1; p > 0; --p) {
            const std::size_t left_idx = 2 * p;
            const std::size_t w_left = (left_idx >= k_) ? (left_idx - k_) : winner[left_idx];

            const std::size_t right_idx = 2 * p + 1;
            const std::size_t w_right = (right_idx >= k_) ? (right_idx - k_) : winner[right_idx];

            if (is_less(w_right, w_left)) {
                tree_[p] = w_left;
                winner[p] = w_right;
            } else {
                tree_[p] = w_right;
                winner[p] = w_left;
            }
        }
        tree_[0] = winner[1];
    }

    /**
     * @brief Returns the stream index of the current global minimum.
     */
    [[nodiscard]] std::size_t top_source() const noexcept {
        return tree_.empty() ? 0 : tree_[0];
    }

    /**
     * @brief Returns a const reference to the current global minimum.
     */
    [[nodiscard]] const ElementType& top() const noexcept {
        return leaves_[top_source()];
    }

    /**
     * @brief Replaces the winning element with a new element from the same stream
     * and replays the tree in strictly ceil(log2(K)) comparisons.
     *
     * @param new_element The replacement element from stream top_source().
     */
    void replace_top(const ElementType& new_element) {
        if (k_ == 0 || active_streams_ == 0) {
            return;
        }
        const std::size_t s = tree_[0];
        leaves_[s] = new_element;
        if (k_ <= 1) {
            return;
        }

        replay(s);
    }

    /**
     * @brief Marks the current winning stream as exhausted/depleted.
     */
    void pop_stream() {
        if (active_streams_ == 0) {
            return;
        }
        const std::size_t s = tree_[0];
        exhausted_[s] = 1;
        --active_streams_;

        if (active_streams_ == 0 || k_ <= 1) {
            return;
        }

        replay(s);
    }

    /**
     * @brief Returns true if all streams are exhausted or if initialized with zero streams.
     */
    [[nodiscard]] bool empty() const noexcept {
        return active_streams_ == 0;
    }

    /**
     * @brief Returns the number of currently active streams.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return active_streams_;
    }

private:
    /**
     * @brief Replays the tournament path from leaf s to root.
     */
    void replay(std::size_t s) {
        std::size_t challenger = s;
        if (active_streams_ == k_) {
            const auto* leaves_data = leaves_.data();
            auto* tree_data = tree_.data();
            const ElementType* challenger_ptr = &leaves_data[challenger];

            for (std::size_t node = (s + k_) >> 1; node > 0; node >>= 1) {
                const std::size_t opponent = tree_data[node];
                const ElementType* opponent_ptr = &leaves_data[opponent];
                const bool opp_wins = comp_(*opponent_ptr, *challenger_ptr);

                tree_data[node] = opp_wins ? challenger : opponent;
                challenger = opp_wins ? opponent : challenger;
                challenger_ptr = opp_wins ? opponent_ptr : challenger_ptr;
            }
        } else {
            // General path handling exhausted streams as sentinels
            for (std::size_t node = (s + k_) >> 1; node > 0; node >>= 1) {
                const std::size_t opponent = tree_[node];
                const bool opp_wins = is_less(opponent, challenger);

                tree_[node] = opp_wins ? challenger : opponent;
                challenger = opp_wins ? opponent : challenger;
            }
        }
        tree_[0] = challenger;
    }

    [[nodiscard]] inline bool is_less(std::size_t a, std::size_t b) const {
        const bool a_ex = (exhausted_[a] != 0);
        const bool b_ex = (exhausted_[b] != 0);
        if (a_ex != b_ex) {
            return b_ex; // Non-exhausted element is strictly less than exhausted sentinel
        }
        if (a_ex) {
            return false;
        }
        return comp_(leaves_[a], leaves_[b]);
    }

    std::size_t k_{0};
    std::size_t active_streams_{0};
    Compare comp_{};

    // Flat contiguous cache-aligned nodes: tree_[i] stores the stream index
    alignas(64) std::vector<std::size_t> tree_;
    std::vector<ElementType> leaves_;
    std::vector<std::uint8_t> exhausted_;
};

/**
 * @brief Merges K sorted ranges into an output iterator using a tournament LoserTree.
 *
 * @tparam RangeOfRanges A range of input ranges conforming to std::ranges::input_range.
 * @tparam Out An output iterator conforming to std::weakly_incrementable.
 * @tparam Comp Comparison functor conforming to strict weak ordering.
 *
 * @param ranges Input collection of presorted ranges.
 * @param out Output destination iterator.
 * @param comp Comparator functor.
 * @return The advanced output iterator.
 */
template <std::ranges::input_range RangeOfRanges, std::weakly_incrementable Out,
          typename Comp = std::ranges::less>
Out k_way_merge(RangeOfRanges&& ranges, Out out, Comp comp = {}) {
    using InnerRef = std::ranges::range_reference_t<RangeOfRanges>;
    using Iter = std::ranges::iterator_t<InnerRef>;
    using Sentinel = std::ranges::sentinel_t<InnerRef>;
    using ValueType = std::iter_value_t<Iter>;

    struct StreamCursor {
        Iter curr;
        Sentinel end;
    };

    std::vector<StreamCursor> streams;
    std::vector<ValueType> heads;

    if constexpr (std::ranges::sized_range<RangeOfRanges>) {
        const std::size_t estimated_k = std::ranges::size(ranges);
        streams.reserve(estimated_k);
        heads.reserve(estimated_k);
    }

    for (auto&& range : ranges) {
        auto it = std::ranges::begin(range);
        auto it_end = std::ranges::end(range);
        if (it != it_end) {
            heads.push_back(*it);
            streams.push_back(StreamCursor{std::move(it), std::move(it_end)});
        }
    }

    const std::size_t k = streams.size();
    if (k == 0) {
        return out;
    }

    LoserTree<ValueType, Comp> tree(k, comp);
    tree.initialize(heads);

    while (!tree.empty()) {
        const std::size_t s = tree.top_source();
        *out = tree.top();
        ++out;

        ++streams[s].curr;
        if (streams[s].curr != streams[s].end) {
            tree.replace_top(*streams[s].curr);
        } else {
            tree.pop_stream();
        }
    }

    return out;
}

} // namespace algoat::sorting
