#include "algoat/searching/eytzinger_search_tree.hpp"
#include "algoat/searching/searching.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::searching;

template <typename Algo> class SearchingPBT : public ::testing::Test {
protected:
    Algo algo;
};

using SearchAlgos =
    ::testing::Types<LinearSearch, BinarySearch, InterpolationSearch, AdaptiveBinarySearch,
                     HybridInterpolationSearch, BranchlessBinarySearch, GallopingSearch>;

TYPED_TEST_SUITE(SearchingPBT, SearchAlgos);

TYPED_TEST(SearchingPBT, Invariants) {
    rc::check("Searching invariants", [this](std::vector<int> data, int target) {
        if (this->algo.requires_sorted()) {
            std::sort(data.begin(), data.end());
        }

        auto result = this->algo.search(std::span<const int>{data}, target);

        if (result.has_value()) {
            RC_ASSERT(data[result.value()] == target);
        } else {
            bool found = false;
            for (auto x : data) {
                if (x == target)
                    found = true;
            }
            RC_ASSERT(!found);
        }
    });
}

TEST(BranchlessBinarySearchPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("BranchlessBinarySearch lower_bound invariants",
              [](std::vector<int> data, int target) {
                  std::sort(data.begin(), data.end());
                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  auto actual_it = branchless_lower_bound(data.begin(), data.end(), target);
                  RC_ASSERT(actual_it == std_it);
              });
}

TEST(GallopingSearchPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("GallopingSearch lower_bound invariants", [](std::vector<int> data, int target) {
        std::sort(data.begin(), data.end());
        auto std_it = std::lower_bound(data.begin(), data.end(), target);
        auto actual_it = galloping_search(data.begin(), data.end(), target);
        RC_ASSERT(actual_it == std_it);
    });
}

TEST(EytzingerSearchPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("EytzingerSearchTree lower_bound invariants", [](std::vector<int> data, int target) {
        std::sort(data.begin(), data.end());
        EytzingerSearchTree<int> tree(std::span<const int>{data});

        auto std_it = std::lower_bound(data.begin(), data.end(), target);
        std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());

        RC_ASSERT(tree.size() == data.size());
        RC_ASSERT(tree.lower_bound(target) == expected_idx);
    });
}
