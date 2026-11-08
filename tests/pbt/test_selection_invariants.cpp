#include "algoat/selection/selection.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

TEST(SelectionPBT, NthElementInvariants) {
    rc::check("nth_element selection invariants", [](std::vector<int> input) {
        if (input.empty()) {
            return;
        }

        auto original = input;
        auto expected = original;
        std::sort(expected.begin(), expected.end());

        std::size_t k = *rc::gen::inRange<std::size_t>(0, input.size());

        algoat::selection::nth_element(input.begin(), input.begin() + k, input.end());

        // 1. Conservation invariant (permutation)
        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        // 2. Exact order statistic value
        RC_ASSERT(input[k] == expected[k]);

        // 3. Selection partition invariant
        for (std::size_t i = 0; i < k; ++i) {
            RC_ASSERT(input[i] <= input[k]);
        }
        for (std::size_t j = k + 1; j < input.size(); ++j) {
            RC_ASSERT(input[k] <= input[j]);
        }
    });
}

TEST(SelectionPBT, PdqSelectInvariants) {
    rc::check("pdqselect selection invariants", [](std::vector<int> input) {
        if (input.empty()) {
            return;
        }

        auto original = input;
        auto expected = original;
        std::sort(expected.begin(), expected.end());

        std::size_t k = *rc::gen::inRange<std::size_t>(0, input.size());

        algoat::selection::pdqselect(input.begin(), input.begin() + k, input.end());

        // 1. Conservation invariant (permutation)
        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        // 2. Exact order statistic value
        RC_ASSERT(input[k] == expected[k]);

        // 3. Selection partition invariant
        for (std::size_t i = 0; i < k; ++i) {
            RC_ASSERT(input[i] <= input[k]);
        }
        for (std::size_t j = k + 1; j < input.size(); ++j) {
            RC_ASSERT(input[k] <= input[j]);
        }
    });
}

TEST(SelectionPBT, SelectOptimalInvariants) {
    rc::check("select_optimal selection invariants", [](std::vector<int> input) {
        if (input.empty()) {
            return;
        }

        auto original = input;
        auto expected = original;
        std::sort(expected.begin(), expected.end());

        std::size_t k = *rc::gen::inRange<std::size_t>(0, input.size());

        algoat::selection::select_optimal(input.begin(), input.begin() + k, input.end());

        // 1. Conservation invariant (permutation)
        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        // 2. Exact order statistic value
        RC_ASSERT(input[k] == expected[k]);

        // 3. Selection partition invariant
        for (std::size_t i = 0; i < k; ++i) {
            RC_ASSERT(input[i] <= input[k]);
        }
        for (std::size_t j = k + 1; j < input.size(); ++j) {
            RC_ASSERT(input[k] <= input[j]);
        }
    });
}

TEST(SelectionPBT, TopKUnsortedInvariants) {
    rc::check("top_k_unsorted invariants", [](std::vector<int> input) {
        auto original = input;
        auto expected = original;
        std::sort(expected.begin(), expected.end());

        std::size_t k = *rc::gen::inRange<std::size_t>(0, input.size() + 5);

        algoat::selection::top_k_unsorted(input, k);

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        if (k > 0 && k < input.size()) {
            auto max_prefix = *std::max_element(input.begin(), input.begin() + k);
            auto min_suffix = *std::min_element(input.begin() + k, input.end());
            RC_ASSERT(max_prefix <= min_suffix);

            auto sorted_prefix = input;
            std::sort(sorted_prefix.begin(), sorted_prefix.begin() + k);
            for (std::size_t i = 0; i < k; ++i) {
                RC_ASSERT(sorted_prefix[i] == expected[i]);
            }
        }
    });
}

TEST(SelectionPBT, TopKSortedInvariants) {
    rc::check("top_k_sorted invariants", [](std::vector<int> input) {
        auto original = input;
        auto expected = original;
        std::sort(expected.begin(), expected.end());

        std::size_t k = *rc::gen::inRange<std::size_t>(0, input.size() + 5);

        algoat::selection::top_k_sorted(input, k);

        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        if (k > 0 && k < input.size()) {
            RC_ASSERT(std::is_sorted(input.begin(), input.begin() + k));
            for (std::size_t i = 0; i < k; ++i) {
                RC_ASSERT(input[i] == expected[i]);
            }
            auto max_prefix = input[k - 1];
            auto min_suffix = *std::min_element(input.begin() + k, input.end());
            RC_ASSERT(max_prefix <= min_suffix);
        } else if (k >= input.size() && !input.empty()) {
            RC_ASSERT(std::is_sorted(input.begin(), input.end()));
        }
    });
}

TEST(SelectionPBT, MultiSelectInvariants) {
    rc::check("multi_select selection invariants", [](std::vector<int> input) {
        if (input.empty()) {
            return;
        }

        auto original = input;
        auto expected = original;
        std::sort(expected.begin(), expected.end());

        // Generate between 1 and min(8, input.size()) sorted ranks
        const std::size_t num_ranks =
            *rc::gen::inRange<std::size_t>(1, std::min<std::size_t>(9, input.size() + 1));
        std::vector<std::size_t> ranks;
        ranks.reserve(num_ranks);
        for (std::size_t i = 0; i < num_ranks; ++i) {
            ranks.push_back(*rc::gen::inRange<std::size_t>(0, input.size()));
        }
        std::sort(ranks.begin(), ranks.end());

        algoat::selection::multi_select(input, ranks);

        // 1. Conservation invariant (permutation)
        RC_ASSERT(input.size() == original.size());
        RC_ASSERT(std::is_permutation(input.begin(), input.end(), original.begin()));

        // 2. Exact order statistic values at each rank
        for (std::size_t r : ranks) {
            RC_ASSERT(input[r] == expected[r]);
        }

        // 3. Prefix bucket invariant before ranks[0]
        for (std::size_t i = 0; i < ranks[0]; ++i) {
            RC_ASSERT(input[i] <= input[ranks[0]]);
        }

        // 4. Intermediate bucket invariants between adjacent ranks
        for (std::size_t r = 0; r + 1 < ranks.size(); ++r) {
            const std::size_t cur = ranks[r];
            const std::size_t nxt = ranks[r + 1];
            RC_ASSERT(input[cur] <= input[nxt]);
            for (std::size_t i = cur + 1; i < nxt; ++i) {
                RC_ASSERT(input[cur] <= input[i]);
                RC_ASSERT(input[i] <= input[nxt]);
            }
        }

        // 5. Suffix bucket invariant after ranks.back()
        for (std::size_t i = ranks.back() + 1; i < input.size(); ++i) {
            RC_ASSERT(input[ranks.back()] <= input[i]);
        }
    });
}

TEST(SelectionPBT, QuantilesInvariants) {
    rc::check("quantiles helper invariants", [](std::vector<int> input) {
        if (input.empty()) {
            return;
        }

        const std::size_t q = *rc::gen::inRange<std::size_t>(2, 12);
        auto copy = input;
        auto result = algoat::selection::quantiles(std::span<int>{copy}, q);

        RC_ASSERT(result.size() == q - 1);
        RC_ASSERT(std::is_sorted(result.begin(), result.end()));

        auto sorted = input;
        std::sort(sorted.begin(), sorted.end());

        for (std::size_t i = 0; i < result.size(); ++i) {
            const std::size_t rank_idx =
                std::clamp(((i + 1) * input.size()) / q, std::size_t{0}, input.size() - 1);
            RC_ASSERT(result[i] == sorted[rank_idx]);
        }
    });
}

TEST(SelectionPBT, DynamicMedianInvariants) {
    rc::check("dynamic stream median invariants", [](std::vector<int> stream) {
        if (stream.empty()) {
            return;
        }

        algoat::selection::DynamicMedian<int> dm;
        std::vector<int> history;

        for (int val : stream) {
            dm.push(val);
            history.push_back(val);

            std::vector<int> sorted = history;
            std::sort(sorted.begin(), sorted.end());
            const std::size_t n = sorted.size();

            double expected = 0.0;
            if (n % 2 != 0) {
                expected = static_cast<double>(sorted[n / 2]);
            } else {
                expected =
                    (static_cast<double>(sorted[n / 2 - 1]) + static_cast<double>(sorted[n / 2])) *
                    0.5;
            }

            RC_ASSERT(dm.size() == n);
            RC_ASSERT(std::abs(dm.median() - expected) < 1e-9);
        }
    });
}

TEST(SelectionPBT, SlidingWindowMedianInvariants) {
    rc::check("sliding window median and order statistics invariants", [](std::vector<int> stream) {
        if (stream.empty()) {
            return;
        }

        const std::size_t w = *rc::gen::inRange<std::size_t>(1, 32);
        algoat::selection::SlidingWindowMedian<int> swm(w);
        std::deque<int> window;

        for (int val : stream) {
            swm.push(val);
            window.push_back(val);
            if (window.size() > w) {
                window.pop_front();
            }

            std::vector<int> sorted(window.begin(), window.end());
            std::sort(sorted.begin(), sorted.end());
            const std::size_t cur_sz = sorted.size();

            double expected = 0.0;
            if (cur_sz % 2 != 0) {
                expected = static_cast<double>(sorted[cur_sz / 2]);
            } else {
                expected = (static_cast<double>(sorted[cur_sz / 2 - 1]) +
                            static_cast<double>(sorted[cur_sz / 2])) *
                           0.5;
            }

            RC_ASSERT(swm.size() == cur_sz);
            RC_ASSERT(std::abs(swm.median() - expected) < 1e-9);

            for (std::size_t r = 0; r < cur_sz; ++r) {
                RC_ASSERT(swm.order_statistic(r) == sorted[r]);
            }
        }
    });
}
