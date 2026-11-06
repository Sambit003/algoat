#include "algoat/sorting/loser_tree.hpp"

#include <algorithm>
#include <concepts>
#include <gtest/gtest.h>
#include <queue>
#include <rapidcheck.h>
#include <vector>

namespace algoat::sorting::testing {
namespace {

// Reference multi-way merge using std::priority_queue
template <typename T> std::vector<T> pq_multiway_merge(const std::vector<std::vector<T>>& runs) {
    struct Element {
        T val;
        std::size_t stream_idx;
        std::size_t elem_idx;

        bool operator>(const Element& other) const {
            return val > other.val;
        }
    };

    std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        if (!runs[i].empty()) {
            pq.push(Element{runs[i][0], i, 0});
        }
    }

    std::vector<T> result;
    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();
        result.push_back(top.val);

        if (top.elem_idx + 1 < runs[top.stream_idx].size()) {
            pq.push(
                Element{runs[top.stream_idx][top.elem_idx + 1], top.stream_idx, top.elem_idx + 1});
        }
    }
    return result;
}

TEST(LoserTreePBT, KWayMergeInvariants) {
    rc::check("k_way_merge preserves monotonicity, multiset conservation, and matches "
              "std::priority_queue",
              [](std::vector<std::vector<int>> runs) {
                  // Ensure each input run is sorted
                  std::vector<int> expected;
                  for (auto& run : runs) {
                      std::sort(run.begin(), run.end());
                      expected.insert(expected.end(), run.begin(), run.end());
                  }
                  std::sort(expected.begin(), expected.end());

                  // 1. LoserTree k_way_merge
                  std::vector<int> actual;
                  k_way_merge(runs, std::back_inserter(actual));

                  // Monotonic sortedness
                  RC_ASSERT(std::is_sorted(actual.begin(), actual.end()));

                  // Size conservation
                  RC_ASSERT(actual.size() == expected.size());

                  // Exact equivalence to sorted concatenation
                  RC_ASSERT(actual == expected);

                  // 2. Differential verification against std::priority_queue
                  auto pq_result = pq_multiway_merge(runs);
                  RC_ASSERT(actual == pq_result);
              });
}

} // namespace
} // namespace algoat::sorting::testing
