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

using SearchAlgos = ::testing::Types<LinearSearch, BinarySearch, InterpolationSearch,
                                     AdaptiveBinarySearch, HybridInterpolationSearch>;

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
