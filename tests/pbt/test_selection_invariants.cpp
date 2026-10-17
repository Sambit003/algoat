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
