#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/interpolation_search.hpp"
#include "algoat/searching/linear_search.hpp"
#include "algoat/searching/searching.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <vector>

using namespace algoat::searching;

template <typename Algo> void verify_searching(Algo& algo, std::vector<int> data, int target) {
    if (algo.requires_sorted()) {
        std::sort(data.begin(), data.end());
    }

    auto result = algo.search(std::span{data}, target);

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
}

// Linear Search
RC_GTEST_PROP(SearchingInvariantsPBT, LinearSearch_Invariants,
              (std::vector<int> data, int target)) {
    LinearSearch algo;
    verify_searching(algo, data, target);
}

// Binary Search
RC_GTEST_PROP(SearchingInvariantsPBT, BinarySearch_Invariants,
              (std::vector<int> data, int target)) {
    BinarySearch algo;
    verify_searching(algo, data, target);
}

// Interpolation Search
RC_GTEST_PROP(SearchingInvariantsPBT, InterpolationSearch_Invariants,
              (std::vector<int> data, int target)) {
    InterpolationSearch algo;
    verify_searching(algo, data, target);
}
